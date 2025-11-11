from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch import nn
import torch
import os
import numpy as np

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

from nnunetv2.nets.LightMUNet import LightMUNet
from torch.optim import Adam

class nnUNetTrainerLightMUNet(nnUNetTrainerNoDeepSupervision):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device('cuda')
        ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.grad_scaler = None
        self.initial_lr = 1e-4
        self.weight_decay = 1e-5
        # epochs: allow override via env for different datasets (e.g., DIA)
        try:
            import os as _os
            self.num_epochs = int(_os.environ.get('NNUNET_EPOCHS', '500'))
        except Exception:
            self.num_epochs = 500
        # Reduce default batch size for 8GB GPUs and allow env override
        try:
            import os as _os
            self.batch_size = int(_os.environ.get('NNUNET_BATCH_SIZE', '2'))
        except Exception:
            self.batch_size = 2

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        import os as _os
        label_manager = plans_manager.get_label_manager(dataset_json)
        try:
            _init_filters = int(_os.environ.get('NNUNET_INIT_FILTERS', '32'))
        except Exception:
            _init_filters = 32
        model = LightMUNet(
            spatial_dims = len(configuration_manager.patch_size),
            init_filters = _init_filters,
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
        )

        return model
    

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        output = self.network(data)
        l = self.loss(output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy()}
    

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        output = self.network(data)
        del data
        l = self.loss(output, target)

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        # Optional validation-time postprocessing: keep largest connected component (binary, non-regions)
        try:
            use_lcc = os.environ.get('NNUNET_USE_LCC_POSTPROC', '0').lower() in ('1', 'true', 'yes')
            if use_lcc and (not self.label_manager.has_regions) and output.shape[1] == 2:
                from scipy import ndimage as ndi
                # work on CPU numpy for connectivity
                pred_fg = predicted_segmentation_onehot[:, 1].detach().cpu().numpy().astype(np.uint8)
                new_fg_list = []
                for b in range(pred_fg.shape[0]):
                    lab, num = ndi.label(pred_fg[b])
                    if num <= 1:
                        new_fg_list.append(pred_fg[b])
                    else:
                        sizes = ndi.sum(pred_fg[b], lab, index=range(1, num + 1))
                        keep_label = int(np.argmax(sizes)) + 1
                        new_fg_list.append((lab == keep_label).astype(np.uint8))
                new_fg = np.stack(new_fg_list, axis=0)
                new_fg_t = torch.from_numpy(new_fg).to(predicted_segmentation_onehot.device, dtype=predicted_segmentation_onehot.dtype)
                predicted_segmentation_onehot[:, 1] = new_fg_t
                predicted_segmentation_onehot[:, 0] = 1 - new_fg_t
        except Exception:
            # best-effort: if scipy not available, skip LCC without failing validation
            pass

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    def configure_optimizers(self):

        optimizer = Adam(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=0.9)

        return optimizer, scheduler
    
    def set_deep_supervision_enabled(self, enabled: bool):
        pass
