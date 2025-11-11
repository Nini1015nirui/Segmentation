#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="nnUNetv2 single-process predictor (no multiprocessing)")
    parser.add_argument('-i', '--input', required=True, help='Input images folder')
    parser.add_argument('-o', '--output', required=True, help='Output folder')
    parser.add_argument('-d', '--dataset', required=True, help='Dataset name or id (e.g., 804 or Dataset804_Foo)')
    parser.add_argument('-c', '--configuration', required=True, help='Configuration (e.g., 2d, 3d_fullres)')
    parser.add_argument('-tr', '--trainer', required=True, help='Trainer class (e.g., nnUNetTrainerLightMUNet)')
    parser.add_argument('-p', '--plans', required=True, help='Plans identifier (e.g., nnUNetPlans or nnUNetPlans_mem4)')
    parser.add_argument('-f', '--fold', default='all', help='Fold to use (e.g., 0..4 or all)')
    parser.add_argument('-chk', '--checkpoint', default='checkpoint_final.pth', help='Checkpoint filename')
    parser.add_argument('--pattern', default=None, help='Optional glob pattern to filter input files (e.g., "*_0000.png")')
    args = parser.parse_args()

    # Resolve nnUNet paths from local repo (lightm-unet)
    # The scripts set PYTHONPATH to prefer the repo version; import after that
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    from nnunetv2.utilities.file_path_utilities import get_output_folder
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

    dataset_name = maybe_convert_to_dataset_name(args.dataset)
    model_folder = get_output_folder(dataset_name, args.trainer, args.plans, args.configuration)

    inp = Path(args.input)
    out = Path(args.output)
    maybe_mkdir_p(out)

    # Collect files
    if args.pattern is None:
        # default: modality suffix _0000.*
        files = sorted([p for p in inp.iterdir() if p.is_file()])
    else:
        files = sorted(inp.glob(args.pattern))

    if not files:
        print(f"No input files found in {inp} (pattern={args.pattern}).")
        return 0

    # Initialize predictor
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=(device.type == 'cuda'),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(args.fold,),
        checkpoint_name=args.checkpoint,
    )

    # Fallback PNG reader to avoid SimpleITK dependency for natural images
    from PIL import Image

    def read_png_as_nnunet_input(pth: Path):
        im = Image.open(pth).convert('L')  # grayscale
        arr = np.array(im, dtype=np.float32)
        # shape -> (b=1, c=1, y, x)
        arr = arr[None, None, ...]
        props = {'spacing': (999.0, 1.0, 1.0)}  # 2D spacing convention in nnUNet
        return arr, props
    for img_path in files:
        # Determine truncated output name
        base = img_path.name
        # Derive output filename. For natural images (e.g., .png) keep the original
        # name to allow downstream overlay scripts to match stems with originals.
        base_no_mod = base
        if not base_no_mod.lower().endswith('.png'):
            # Strip modality suffix like _0000 only for nnU-Net style inputs
            if '_0000' in base_no_mod:
                base_no_mod = base_no_mod.replace('_0000', '')
        # ensure .png extension for output
        if not base_no_mod.lower().endswith('.png'):
            base_no_mod = Path(base_no_mod).with_suffix('.png').name
        ofile = out / base_no_mod

        # Read image and predict (PNG fallback reader)
        img, props = read_png_as_nnunet_input(img_path)
        seg = predictor.predict_single_npy_array(
            img, props, None, output_file_truncated=None, save_or_return_probabilities=False
        )
        # Normalize shape to 2D (H, W): take the last two dims as image axes
        seg_arr = np.asarray(seg)
        if seg_arr.ndim >= 2:
            H, W = seg_arr.shape[-2], seg_arr.shape[-1]
            seg_arr = seg_arr.reshape(-1, H, W)[0]
        else:
            # Fallback: treat as a row image
            seg_arr = seg_arr.reshape(1, -1)
        # Save as label/binary PNG (0/255 for binary)
        seg_u8 = seg_arr.astype(np.uint8)
        if seg_u8.max() <= 1:
            seg_u8 = seg_u8 * 255
        seg_img = Image.fromarray(seg_u8)
        seg_img.save(ofile)
        print(f"Saved prediction for {img_path.name} -> {ofile}")

    print("Done (no-mp prediction)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
