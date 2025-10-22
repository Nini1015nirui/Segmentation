# 推理与可视化完整指南

## 📋 概述

本指南介绍如何使用训练好的 LightM-UNet 模型进行推理、应用 nnU-Net 后处理，并可视化分割结果。

---

## 🎯 快速开始（一键执行）

### 完整流程：后处理 + 可视化

```bash
# 一键完成：后处理预测 + 可视化对比
bash scripts/run_postprocess_and_visualize.sh

# 自定义可视化样本数
bash scripts/run_postprocess_and_visualize.sh 803 2d nnUNetTrainerLightMUNet 20

# 参数说明：
# $1 - 数据集ID（默认：803）
# $2 - 配置（默认：2d）
# $3 - 训练器（默认：nnUNetTrainerLightMUNet）
# $4 - 可视化样本数（默认：10）
```

**执行内容**：
1. ✅ 对训练集做 TTA 预测（用于确定后处理参数）
2. ✅ 自动确定最优后处理策略（生成 `postprocessing.pkl`）
3. ✅ 对测试集做 TTA 预测
4. ✅ 应用后处理到预测结果
5. ✅ 生成可视化对比图（原始 vs 后处理 vs 真实标签）
6. ✅ 计算性能指标统计

---

## 📊 分步骤执行

如果你想分步骤理解和执行流程，可以按以下步骤操作。

### Step 1: 运行后处理预测

```bash
bash scripts/apply_tta_and_postprocessing.sh 803 2d nnUNetTrainerLightMUNet
```

**输出文件**：
- `data/nnUNet_results/Dataset803_TA/nnUNetTrainerLightMUNet__nnUNetPlans__2d/val_for_pp/`
  - `postprocessing.pkl` - 自动确定的后处理参数
- `data/test_predictions/tta_on/` - 原始 TTA 预测结果
- `data/test_predictions/tta_on_pp/` - 后处理后的预测结果

### Step 2: 可视化对比

```bash
python scripts/visualize_with_postprocessing.py \
    --pred_raw_dir data/test_predictions/tta_on \
    --pred_pp_dir data/test_predictions/tta_on_pp \
    --label_dir data/nnUNet_raw/Dataset803_TA/labelsTr \
    --image_dir data/nnUNet_raw/Dataset803_TA/imagesTr \
    --output_dir data/postprocess_visualization \
    --num_samples 10
```

**输出文件**：
- `data/postprocess_visualization/visualizations/` - 可视化对比图
- `data/postprocess_visualization/postprocess_summary.json` - 统计结果JSON

---

## 🖼️ 可视化输出说明

### 可视化图片内容

每张可视化图片包含 **10 个子图**，分为两行：

#### 第一行：总体对比
1. **Original Image** - 原始医学影像
2. **Ground Truth** - 真实标签（红色覆盖）
3. **Raw Prediction** - 原始预测（蓝色覆盖）+ Dice 分数
4. **Post-processed** - 后处理预测（绿色覆盖）+ Dice 分数和提升值
5. **Error Analysis** - 错误分析图
   - 🟢 绿色：真阳性（TP）
   - 🔵 蓝色：假阳性（FP）
   - 🔴 红色：假阴性（FN）

#### 第二行：边界细节对比
6. **Boundary: GT vs Raw** - GT（红）vs 原始预测（蓝）边界对比
   - 紫色区域 = 重叠部分
7. **Boundary: GT vs Post** - GT（红）vs 后处理（绿）边界对比
   - 黄色区域 = 重叠部分
8. **Raw Boundary Detail** - 原始预测边界细节
9. **Post-processed Boundary Detail** - 后处理边界细节
10. **Metrics Comparison** - 指标对比表格
    - Dice, IoU, Precision, Recall
    - 原始 vs 后处理 vs 提升值

### 统计JSON内容

```json
{
  "timestamp": "2025-10-03T...",
  "num_samples": 10,
  "postprocess_method": "nnUNet Official (nnUNetv2_determine_postprocessing)",
  "statistics": {
    "raw_prediction": {
      "dice": 0.8542,
      "iou": 0.7456,
      "precision": 0.8621,
      "recall": 0.8467
    },
    "postprocessed": {
      "dice": 0.8687,
      "iou": 0.7689,
      "precision": 0.8754,
      "recall": 0.8623
    },
    "improvement": {
      "avg_dice_gain": 0.0145,
      "avg_iou_gain": 0.0233,
      "improved_cases": 8,
      "improvement_rate": "80.0%"
    }
  },
  "detailed_results": [...]
}
```

---

## 🔧 自定义使用

### 仅运行后处理（不可视化）

```bash
bash scripts/apply_tta_and_postprocessing.sh 803 2d nnUNetTrainerLightMUNet
```

### 仅可视化已有预测结果

```bash
python scripts/visualize_with_postprocessing.py \
    --pred_raw_dir /path/to/raw_predictions \
    --pred_pp_dir /path/to/postprocessed_predictions \
    --label_dir /path/to/labels \
    --image_dir /path/to/images \
    --output_dir /path/to/output \
    --num_samples 20
```

### 对不同数据集运行

```bash
# 对 3D 配置运行
bash scripts/run_postprocess_and_visualize.sh 803 3d_fullres nnUNetTrainerLightMUNet 15

# 对其他数据集运行
bash scripts/run_postprocess_and_visualize.sh 101 2d nnUNetTrainerLightMUNet 10
```

---

## 📈 后处理效果评估

### 后处理策略类型

nnU-Net 自动确定后处理可能应用以下策略：

1. **移除小连通分量**
   - 去除面积小于阈值的预测区域
   - 减少假阳性噪声

2. **保留最大连通分量（LCC）**
   - 仅保留最大的连通区域
   - 适用于单目标分割

3. **无后处理**
   - 如果后处理降低性能，则不应用

### 性能指标说明

| 指标 | 说明 | 优化目标 |
|-----|------|---------|
| **Dice** | 预测与GT的重叠度 | 越高越好（0-1） |
| **IoU** | 交并比 | 越高越好（0-1） |
| **Precision** | 预测中真阳性的比例 | 越高越好（0-1） |
| **Recall** | GT中被检测出的比例 | 越高越好（0-1） |

### 典型提升幅度

根据 nnU-Net 官方文档和实践经验：

| 后处理组件 | 典型 Dice 提升 | 时间开销 |
|-----------|--------------|---------|
| **TTA（镜像）** | +0.5% - 2.0% | 2x 推理时间 |
| **自动后处理** | +0.2% - 1.0% | <10% 推理时间 |
| **总计** | +0.7% - 3.0% | ~2.2x 推理时间 |

---

## 🗂️ 输出文件结构

```
data/
├── test_predictions/
│   ├── tta_on/                    # 原始TTA预测
│   │   ├── case_001.nii.gz
│   │   ├── case_002.nii.gz
│   │   └── ...
│   └── tta_on_pp/                 # 后处理后预测
│       ├── case_001.nii.gz
│       ├── case_002.nii.gz
│       └── ...
├── postprocess_visualization/     # 可视化输出
│   ├── visualizations/            # 可视化图片
│   │   ├── case_001_postprocess_comparison.png
│   │   ├── case_002_postprocess_comparison.png
│   │   └── ...
│   └── postprocess_summary.json   # 统计结果
└── nnUNet_results/
    └── Dataset803_TA/
        └── nnUNetTrainerLightMUNet__nnUNetPlans__2d/
            └── val_for_pp/
                └── postprocessing.pkl  # 后处理参数
```

---

## 🛠️ 故障排查

### 问题1：未找到模型文件

**错误**：
```
❌ 未找到模型目录: .../nnUNetTrainerLightMUNet__nnUNetPlans__2d
```

**解决方法**：
```bash
# 检查训练是否完成
ls -lh data/nnUNet_results/Dataset803_TA/nnUNetTrainerLightMUNet__nnUNetPlans__2d/fold_all/*.pth

# 确认有 checkpoint_best.pth 文件
```

### 问题2：后处理确定失败

**错误**：
```
❌ 确定后处理失败
```

**解决方法**：
```bash
# 检查 imagesTr 和 labelsTr 是否存在且一一对应
ls -lh data/nnUNet_raw/Dataset803_TA/imagesTr/ | wc -l
ls -lh data/nnUNet_raw/Dataset803_TA/labelsTr/ | wc -l

# 检查是否有足够的磁盘空间
df -h
```

### 问题3：可视化时找不到文件

**错误**：
```
⚠️  跳过 case_xxx: 未找到原始预测
```

**解决方法**：
```bash
# 确保先运行后处理脚本
bash scripts/apply_tta_and_postprocessing.sh

# 检查预测文件是否存在
ls -lh data/test_predictions/tta_on/
ls -lh data/test_predictions/tta_on_pp/
```

### 问题4：CUDA 内存不足

**错误**：
```
RuntimeError: CUDA out of memory
```

**解决方法**：
```bash
# 减少并行进程数
export NNUNET_NP=4  # 默认是8

# 或者分批处理测试集
bash scripts/run_postprocess_and_visualize.sh 803 2d nnUNetTrainerLightMUNet 5
```

---

## 💡 使用建议

### 1. 推理流程建议

**推荐顺序**：
1. ✅ 训练完成后，先运行完整的后处理流程
2. ✅ 查看可视化结果，评估后处理效果
3. ✅ 如果后处理提升明显，则在最终提交中使用后处理结果
4. ✅ 如果后处理反而降低性能，检查数据特点（如是否有多个分离目标）

### 2. 可视化样本选择

```bash
# 快速检查（5个样本）
bash scripts/run_postprocess_and_visualize.sh 803 2d nnUNetTrainerLightMUNet 5

# 详细评估（20个样本）
bash scripts/run_postprocess_and_visualize.sh 803 2d nnUNetTrainerLightMUNet 20

# 全数据集评估（可能需要较长时间）
bash scripts/run_postprocess_and_visualize.sh 803 2d nnUNetTrainerLightMUNet 100
```

### 3. 后处理参数复用

生成的 `postprocessing.pkl` 可以直接用于新数据：

```bash
# 对新数据应用已有后处理参数
nnUNetv2_predict -i /path/to/new_data -o /path/to/output \
  -d 803 -c 2d -tr nnUNetTrainerLightMUNet -f all \
  -chk checkpoint_best.pth

nnUNetv2_apply_postprocessing \
  -i /path/to/output \
  -o /path/to/output_pp \
  -pp_pkl_file data/nnUNet_results/Dataset803_TA/.../val_for_pp/postprocessing.pkl \
  -np 8 \
  -plans_json data/nnUNet_results/Dataset803_TA/.../plans.json \
  -dataset_json data/nnUNet_results/Dataset803_TA/.../dataset.json
```

---

## 📚 相关文档

- [后处理详细指南](postprocessing_guide.md) - 后处理原理和配置
- [训练脚本说明](../scripts/start_training.sh) - 训练时LCC后处理
- [nnU-Net 官方文档](https://github.com/MIC-DKFZ/nnUNet) - 官方参考文档

---

## ✅ 检查清单

使用本指南前，请确认：

- [x] 训练已完成，存在 `checkpoint_best.pth`
- [ ] 运行一键脚本：`bash scripts/run_postprocess_and_visualize.sh`
- [ ] 检查输出目录中的可视化图片
- [ ] 查看 `postprocess_summary.json` 中的统计结果
- [ ] 评估后处理是否提升了性能
- [ ] 如有提升，使用 `tta_on_pp/` 作为最终预测结果

---

**最后更新**：2025-10-03
**版本**：1.0
