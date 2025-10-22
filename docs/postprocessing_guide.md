# nnU-Net 后处理指南

## 📋 概述

LightM-UNet 项目集成了 nnU-Net 的两种后处理机制：
1. **训练时后处理**（验证阶段）：LCC（最大连通分量保留）
2. **推理时后处理**：自动确定和应用后处理参数

---

## 🔧 训练时后处理 (LCC)

### 功能说明
- **名称**：Largest Connected Component (LCC)
- **作用**：在验证阶段自动保留最大的连通区域，去除小的离散预测
- **适用场景**：二分类分割任务（前景/背景）
- **实现位置**：`lightm-unet/nnunetv2/training/nnUNetTrainer/nnUNetTrainerLightMUNet.py:100-122`

### 启用方法
已在 `scripts/start_training.sh` 中启用：
```bash
export NNUNET_USE_LCC_POSTPROC="1"
```

### 工作机制
```python
# 验证步骤中的后处理逻辑
if use_lcc and (not self.label_manager.has_regions) and output.shape[1] == 2:
    # 1. 提取前景预测
    pred_fg = predicted_segmentation_onehot[:, 1].cpu().numpy()

    # 2. 对每个样本进行连通分量分析
    for b in range(pred_fg.shape[0]):
        lab, num = ndi.label(pred_fg[b])  # 标记所有连通区域

        # 3. 保留最大连通区域
        if num > 1:
            sizes = ndi.sum(pred_fg[b], lab, index=range(1, num + 1))
            keep_label = int(np.argmax(sizes)) + 1
            new_fg = (lab == keep_label).astype(np.uint8)
```

### 效果
- ✅ 减少假阳性（去除小的错误预测）
- ✅ 提高验证集 Dice 指标
- ⚠️ 依赖 scipy 库（已在环境中安装）

---

## 🎯 推理时后处理

### 完整流程
使用 `scripts/apply_tta_and_postprocessing.sh` 实现全自动后处理：

#### Step 1: 生成后处理参数
```bash
# 对训练集做预测，用于确定最优后处理参数
nnUNetv2_predict -i imagesTr -o val_for_pp \
  -d 803 -c 2d -tr nnUNetTrainerLightMUNet \
  -f all -chk checkpoint_best.pth
```

#### Step 2: 自动确定后处理
```bash
# 基于训练集预测和真实标签，自动确定后处理策略
nnUNetv2_determine_postprocessing \
  -i val_for_pp \
  -ref labelsTr \
  -p plans.json \
  -d dataset.json \
  -np 8
```

生成文件：`val_for_pp/postprocessing.pkl`

#### Step 3: TTA 预测
```bash
# 带测试时增强的预测（默认启用镜像翻转）
nnUNetv2_predict -i imagesTs -o predictions_tta \
  -d 803 -c 2d -tr nnUNetTrainerLightMUNet \
  -f all -chk checkpoint_best.pth
# 注意：不加 --disable_tta 即启用 TTA
```

#### Step 4: 应用后处理
```bash
# 将自动确定的后处理策略应用到预测结果
nnUNetv2_apply_postprocessing \
  -i predictions_tta \
  -o predictions_tta_pp \
  -pp_pkl_file val_for_pp/postprocessing.pkl \
  -np 8 \
  -plans_json plans.json \
  -dataset_json dataset.json
```

### 一键执行
```bash
# 自动完成上述所有步骤
bash scripts/apply_tta_and_postprocessing.sh 803 2d nnUNetTrainerLightMUNet

# 输出目录
# data/test_predictions/tta_on      - TTA预测结果
# data/test_predictions/tta_on_pp   - 后处理后的最终结果
```

---

## 📊 后处理策略说明

### 自动确定的后处理类型
nnU-Net 可能自动启用以下后处理策略：

1. **移除小连通分量**
   - 去除面积小于阈值的预测区域
   - 阈值基于训练集统计自动确定

2. **保留最大连通分量**
   - 仅保留最大的连通区域
   - 适用于单目标分割（如单个器官）

3. **无后处理**
   - 如果后处理降低性能，则不应用

### 后处理效果评估
`nnUNetv2_determine_postprocessing` 会：
- 测试不同后处理策略
- 计算每种策略的 Dice 分数
- 自动选择最优策略
- 保存到 `postprocessing.pkl`

---

## 🔍 使用建议

### 训练时 LCC
- ✅ **推荐启用**：适用于大部分二分类分割任务
- ✅ **场景**：目标是单个完整区域（如肿瘤、器官）
- ❌ **不适用**：多个分离目标（如多个病灶）

### 推理时后处理
- ✅ **必须使用**：获得最佳预测质量
- ✅ **TTA**：通常提升 0.5-2% Dice
- ✅ **自动后处理**：根据数据特点自适应优化

### 性能影响
| 操作 | 时间增加 | 性能提升 |
|------|---------|---------|
| TTA（镜像） | ~2倍 | +0.5-2% Dice |
| 后处理应用 | ~10% | +0.2-1% Dice |
| 总计 | ~2.2倍 | +0.7-3% Dice |

---

## 🛠️ 故障排查

### 问题1：scipy 导入失败
```bash
# 解决方法
conda activate seg
pip install scipy
```

### 问题2：未找到 postprocessing.pkl
```bash
# 确保先运行后处理确定步骤
nnUNetv2_determine_postprocessing -i val_for_pp -ref labelsTr ...
```

### 问题3：后处理反而降低性能
- 检查数据集是否包含多个分离目标
- 考虑禁用 LCC 后处理（设置 `NNUNET_USE_LCC_POSTPROC="0"`）
- 查看 `postprocessing.pkl` 日志，确认选择的策略

---

## 📁 相关文件

| 文件 | 说明 |
|-----|------|
| `scripts/start_training.sh` | 训练脚本（已启用 LCC） |
| `scripts/apply_tta_and_postprocessing.sh` | 推理后处理脚本 |
| `lightm-unet/nnunetv2/training/nnUNetTrainer/nnUNetTrainerLightMUNet.py:100-122` | LCC 实现代码 |
| `data/test_predictions/tta_on_pp/` | 最终后处理结果输出目录 |

---

## ✅ 快速检查清单

- [x] 训练时 LCC 已在 `start_training.sh` 中启用
- [ ] 训练完成后，运行 `apply_tta_and_postprocessing.sh`
- [ ] 检查输出目录是否生成 `postprocessing.pkl`
- [ ] 对比后处理前后的预测质量

---

**最后更新**：2025-10-03
