#!/usr/bin/env python3
"""
LightM-UNet 后处理可视化脚本
对比原始预测、官方后处理预测、真实标签的效果
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from datetime import datetime
from tqdm import tqdm
import argparse


class PostprocessingVisualizer:
    """后处理可视化器"""

    def __init__(self, pred_raw_dir, pred_pp_dir, label_dir, image_dir, output_dir, num_samples=10):
        self.pred_raw_dir = Path(pred_raw_dir)
        self.pred_pp_dir = Path(pred_pp_dir)
        self.label_dir = Path(label_dir)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)

        # 结果存储
        self.results = []

    def calculate_metrics(self, pred, gt):
        """计算分割指标"""
        pred_bin = (pred > 0).astype(np.uint8)
        gt_bin = (gt > 0).astype(np.uint8)

        intersection = np.sum(pred_bin & gt_bin)
        union = np.sum(pred_bin | gt_bin)
        pred_sum = np.sum(pred_bin)
        gt_sum = np.sum(gt_bin)

        dice = (2.0 * intersection) / (pred_sum + gt_sum + 1e-8)
        iou = intersection / (union + 1e-8)
        precision = intersection / (pred_sum + 1e-8)
        recall = intersection / (gt_sum + 1e-8)

        return {
            'dice': float(dice),
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall)
        }

    def extract_boundary(self, mask, thickness=2):
        """提取边界"""
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        kernel = np.ones((thickness, thickness), np.uint8)
        eroded = cv2.erode(mask_uint8, kernel, iterations=1)
        boundary = mask_uint8 - eroded
        return boundary

    def visualize_comparison(self, image, gt, pred_raw, pred_post, metrics_raw, metrics_post, save_path, case_name):
        """生成三方对比可视化"""
        fig = plt.figure(figsize=(24, 12))

        # 确保数据类型正确
        gt = (gt > 0).astype(np.uint8)
        pred_raw = (pred_raw > 0).astype(np.uint8)
        pred_post = (pred_post > 0).astype(np.uint8)

        # 第一行：原图、GT、原始预测、后处理预测
        ax1 = plt.subplot(2, 5, 1)
        ax1.imshow(image, cmap='gray')
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')

        ax2 = plt.subplot(2, 5, 2)
        ax2.imshow(image, cmap='gray')
        ax2.imshow(gt, cmap='Reds', alpha=0.5)
        ax2.set_title('Ground Truth', fontsize=14, fontweight='bold')
        ax2.axis('off')

        ax3 = plt.subplot(2, 5, 3)
        ax3.imshow(image, cmap='gray')
        ax3.imshow(pred_raw, cmap='Blues', alpha=0.5)
        dice_raw = metrics_raw['dice']
        ax3.set_title(f'Raw Prediction\nDice: {dice_raw:.4f}', fontsize=14, fontweight='bold')
        ax3.axis('off')

        ax4 = plt.subplot(2, 5, 4)
        ax4.imshow(image, cmap='gray')
        ax4.imshow(pred_post, cmap='Greens', alpha=0.5)
        dice_post = metrics_post['dice']
        improvement = dice_post - dice_raw
        color = 'green' if improvement > 0 else 'red'
        ax4.set_title(f'Post-processed (nnU-Net)\nDice: {dice_post:.4f} ({improvement:+.4f})',
                     fontsize=14, fontweight='bold', color=color)
        ax4.axis('off')

        # 差异图
        ax5 = plt.subplot(2, 5, 5)
        diff_map = np.zeros((*image.shape, 3), dtype=np.uint8)
        # 假阳性：原始预测有但GT没有（蓝色）
        fp_raw = (pred_raw > 0) & (gt == 0)
        fp_post = (pred_post > 0) & (gt == 0)
        # 假阴性：GT有但预测没有（红色）
        fn_raw = (gt > 0) & (pred_raw == 0)
        fn_post = (gt > 0) & (pred_post == 0)
        # 真阳性：预测和GT都有（绿色）
        tp_post = (pred_post > 0) & (gt > 0)

        diff_map[fp_post > 0] = [0, 0, 255]   # 假阳性-蓝色
        diff_map[fn_post > 0] = [255, 0, 0]   # 假阴性-红色
        diff_map[tp_post > 0] = [0, 255, 0]   # 真阳性-绿色

        ax5.imshow(diff_map)
        ax5.set_title('Error Analysis\nTP(Green) FP(Blue) FN(Red)', fontsize=14, fontweight='bold')
        ax5.axis('off')

        # 第二行：边界对比
        ax6 = plt.subplot(2, 5, 6)
        boundary_gt = self.extract_boundary(gt, thickness=3)
        boundary_raw = self.extract_boundary(pred_raw, thickness=3)
        overlay = np.zeros((*image.shape, 3), dtype=np.float32)
        overlay[boundary_gt > 0] += [255, 0, 0]    # GT红色
        overlay[boundary_raw > 0] += [0, 0, 255]   # 预测蓝色
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        ax6.imshow(image, cmap='gray')
        ax6.imshow(overlay, alpha=0.8)
        ax6.set_title('Boundary: GT(Red) vs Raw(Blue)\nOverlap=Purple', fontsize=12, fontweight='bold')
        ax6.axis('off')

        ax7 = plt.subplot(2, 5, 7)
        boundary_post = self.extract_boundary(pred_post, thickness=3)
        overlay2 = np.zeros((*image.shape, 3), dtype=np.float32)
        overlay2[boundary_gt > 0] += [255, 0, 0]     # GT红色
        overlay2[boundary_post > 0] += [0, 255, 0]   # 后处理绿色
        overlay2 = np.clip(overlay2, 0, 255).astype(np.uint8)
        ax7.imshow(image, cmap='gray')
        ax7.imshow(overlay2, alpha=0.8)
        ax7.set_title('Boundary: GT(Red) vs Post(Green)\nOverlap=Yellow', fontsize=12, fontweight='bold')
        ax7.axis('off')

        # 边界细节对比
        ax8 = plt.subplot(2, 5, 8)
        ax8.imshow(boundary_raw, cmap='Blues')
        ax8.set_title('Raw Boundary Detail', fontsize=12, fontweight='bold')
        ax8.axis('off')

        ax9 = plt.subplot(2, 5, 9)
        ax9.imshow(boundary_post, cmap='Greens')
        ax9.set_title('Post-processed Boundary Detail', fontsize=12, fontweight='bold')
        ax9.axis('off')

        # 指标对比表
        ax10 = plt.subplot(2, 5, 10)
        ax10.axis('off')
        metrics_text = f"""
        Metrics Comparison

        {"Metric":<12} {"Raw":<10} {"Post-proc":<10} {"Δ":<10}
        {"─"*50}
        {"Dice":<12} {metrics_raw['dice']:<10.4f} {metrics_post['dice']:<10.4f} {metrics_post['dice']-metrics_raw['dice']:+.4f}
        {"IoU":<12} {metrics_raw['iou']:<10.4f} {metrics_post['iou']:<10.4f} {metrics_post['iou']-metrics_raw['iou']:+.4f}
        {"Precision":<12} {metrics_raw['precision']:<10.4f} {metrics_post['precision']:<10.4f} {metrics_post['precision']-metrics_raw['precision']:+.4f}
        {"Recall":<12} {metrics_raw['recall']:<10.4f} {metrics_post['recall']:<10.4f} {metrics_post['recall']-metrics_raw['recall']:+.4f}
        """
        ax10.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                 verticalalignment='center')

        plt.suptitle(f'Case: {case_name} - Postprocessing Effect Comparison',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def process_dataset(self):
        """处理数据集"""
        print(f"\n🚀 开始处理后处理可视化...")

        # 获取预测文件列表（基于后处理目录）
        pred_files = sorted(list(self.pred_pp_dir.glob("*.nii.gz")))[:self.num_samples]

        if not pred_files:
            # 尝试 .png 格式
            pred_files = sorted(list(self.pred_pp_dir.glob("*.png")))[:self.num_samples]

        if not pred_files:
            print(f"❌ 在 {self.pred_pp_dir} 中未找到预测文件")
            return

        print(f"📁 找到 {len(pred_files)} 个后处理预测文件")

        for idx, pred_pp_path in enumerate(tqdm(pred_files, desc="可视化进度")):
            # 提取case名称
            case_name = pred_pp_path.stem
            if case_name.endswith('.nii'):
                case_name = case_name[:-4]

            # 查找对应文件
            # 原始预测
            pred_raw_path = self.pred_raw_dir / pred_pp_path.name
            # 标签（可能需要去除_0000后缀）
            label_name = case_name.replace('_0000', '')
            label_path = self.label_dir / f"{label_name}.nii.gz"
            if not label_path.exists():
                label_path = self.label_dir / f"{label_name}.png"
            # 原图
            image_path = self.image_dir / f"{label_name}_0000.nii.gz"
            if not image_path.exists():
                image_path = self.image_dir / f"{label_name}_0000.png"

            # 验证文件存在
            if not pred_raw_path.exists():
                print(f"⚠️  跳过 {case_name}: 未找到原始预测 {pred_raw_path}")
                continue
            if not label_path.exists():
                print(f"⚠️  跳过 {case_name}: 未找到标签 {label_path}")
                continue
            if not image_path.exists():
                print(f"⚠️  跳过 {case_name}: 未找到原图 {image_path}")
                continue

            # 读取文件
            try:
                if pred_pp_path.suffix == '.gz':
                    # NIfTI 格式
                    import SimpleITK as sitk
                    image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))[0]
                    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))[0]
                    pred_raw = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_raw_path)))[0]
                    pred_post = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_pp_path)))[0]
                else:
                    # PNG 格式
                    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                    gt = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
                    pred_raw = cv2.imread(str(pred_raw_path), cv2.IMREAD_GRAYSCALE)
                    pred_post = cv2.imread(str(pred_pp_path), cv2.IMREAD_GRAYSCALE)

                # 计算指标
                metrics_raw = self.calculate_metrics(pred_raw, gt)
                metrics_post = self.calculate_metrics(pred_post, gt)

                # 可视化
                vis_path = self.output_dir / "visualizations" / f"{case_name}_postprocess_comparison.png"
                self.visualize_comparison(
                    image, gt, pred_raw, pred_post,
                    metrics_raw, metrics_post, vis_path, case_name
                )

                # 保存结果
                result = {
                    'case': case_name,
                    'metrics_raw': metrics_raw,
                    'metrics_post': metrics_post,
                    'improvement': {
                        'dice': metrics_post['dice'] - metrics_raw['dice'],
                        'iou': metrics_post['iou'] - metrics_raw['iou'],
                        'precision': metrics_post['precision'] - metrics_raw['precision'],
                        'recall': metrics_post['recall'] - metrics_raw['recall']
                    }
                }
                self.results.append(result)

            except Exception as e:
                print(f"⚠️  处理 {case_name} 时出错: {e}")
                continue

        # 保存统计结果
        self.save_summary()

    def save_summary(self):
        """保存统计摘要"""
        if not self.results:
            print("❌ 没有结果可保存")
            return

        # 计算平均指标
        avg_raw = {
            'dice': np.mean([r['metrics_raw']['dice'] for r in self.results]),
            'iou': np.mean([r['metrics_raw']['iou'] for r in self.results]),
            'precision': np.mean([r['metrics_raw']['precision'] for r in self.results]),
            'recall': np.mean([r['metrics_raw']['recall'] for r in self.results]),
        }

        avg_post = {
            'dice': np.mean([r['metrics_post']['dice'] for r in self.results]),
            'iou': np.mean([r['metrics_post']['iou'] for r in self.results]),
            'precision': np.mean([r['metrics_post']['precision'] for r in self.results]),
            'recall': np.mean([r['metrics_post']['recall'] for r in self.results]),
        }

        improvement_count = sum(1 for r in self.results if r['improvement']['dice'] > 0)
        degradation_count = sum(1 for r in self.results if r['improvement']['dice'] < 0)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(self.results),
            'postprocess_method': 'nnUNet Official (nnUNetv2_determine_postprocessing)',
            'statistics': {
                'raw_prediction': avg_raw,
                'postprocessed': avg_post,
                'improvement': {
                    'avg_dice_gain': avg_post['dice'] - avg_raw['dice'],
                    'avg_iou_gain': avg_post['iou'] - avg_raw['iou'],
                    'avg_precision_gain': avg_post['precision'] - avg_raw['precision'],
                    'avg_recall_gain': avg_post['recall'] - avg_raw['recall'],
                    'improved_cases': improvement_count,
                    'degraded_cases': degradation_count,
                    'unchanged_cases': len(self.results) - improvement_count - degradation_count,
                    'improvement_rate': f"{100 * improvement_count / len(self.results):.1f}%"
                }
            },
            'detailed_results': self.results
        }

        # 保存JSON
        summary_path = self.output_dir / "postprocess_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 打印摘要
        print("\n" + "="*80)
        print("📊 nnU-Net 后处理效果统计")
        print("="*80)
        print(f"\n样本数量: {len(self.results)}")
        print(f"后处理方法: nnUNet 官方自动确定后处理")
        print(f"\n{'指标':<15} {'原始预测':<15} {'后处理后':<15} {'提升':<15}")
        print("─"*60)
        print(f"{'Dice':<15} {avg_raw['dice']:<15.4f} {avg_post['dice']:<15.4f} {avg_post['dice']-avg_raw['dice']:+.4f}")
        print(f"{'IoU':<15} {avg_raw['iou']:<15.4f} {avg_post['iou']:<15.4f} {avg_post['iou']-avg_raw['iou']:+.4f}")
        print(f"{'Precision':<15} {avg_raw['precision']:<15.4f} {avg_post['precision']:<15.4f} {avg_post['precision']-avg_raw['precision']:+.4f}")
        print(f"{'Recall':<15} {avg_raw['recall']:<15.4f} {avg_post['recall']:<15.4f} {avg_post['recall']-avg_raw['recall']:+.4f}")
        print(f"\n改善情况:")
        print(f"  ✅ 提升案例: {improvement_count}/{len(self.results)} ({100*improvement_count/len(self.results):.1f}%)")
        print(f"  ❌ 降低案例: {degradation_count}/{len(self.results)} ({100*degradation_count/len(self.results):.1f}%)")
        print(f"  ➖ 无变化案例: {len(self.results) - improvement_count - degradation_count}/{len(self.results)}")
        print(f"\n平均Dice提升: {avg_post['dice']-avg_raw['dice']:+.4f} ({100*(avg_post['dice']-avg_raw['dice'])/avg_raw['dice']:+.2f}%)")
        print(f"\n结果保存位置:")
        print(f"  📊 统计JSON: {summary_path}")
        print(f"  🖼️  可视化图片: {self.output_dir / 'visualizations'}")
        print("="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LightM-UNet 后处理效果可视化')
    parser.add_argument('--pred_raw_dir', type=str,
                       default='/mnt/e/LightM-UNet-master/data/test_predictions/tta_on',
                       help='原始预测目录')
    parser.add_argument('--pred_pp_dir', type=str,
                       default='/mnt/e/LightM-UNet-master/data/test_predictions/tta_on_pp',
                       help='后处理预测目录')
    parser.add_argument('--label_dir', type=str,
                       default='/mnt/e/LightM-UNet-master/data/nnUNet_raw/Dataset803_TA/labelsTr',
                       help='标签目录')
    parser.add_argument('--image_dir', type=str,
                       default='/mnt/e/LightM-UNet-master/data/nnUNet_raw/Dataset803_TA/imagesTr',
                       help='原图目录')
    parser.add_argument('--output_dir', type=str,
                       default='/mnt/e/LightM-UNet-master/data/postprocess_visualization',
                       help='输出目录')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='可视化样本数量')

    args = parser.parse_args()

    print("="*80)
    print("🎯 LightM-UNet 后处理效果可视化")
    print("="*80)
    print(f"原始预测: {args.pred_raw_dir}")
    print(f"后处理预测: {args.pred_pp_dir}")
    print(f"标签目录: {args.label_dir}")
    print(f"原图目录: {args.image_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"处理样本: {args.num_samples}")
    print("="*80)

    # 创建可视化器并运行
    visualizer = PostprocessingVisualizer(
        pred_raw_dir=args.pred_raw_dir,
        pred_pp_dir=args.pred_pp_dir,
        label_dir=args.label_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )

    visualizer.process_dataset()

    print("\n✅ 可视化完成！")


if __name__ == '__main__':
    main()
