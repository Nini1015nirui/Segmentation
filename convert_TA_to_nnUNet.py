#!/usr/bin/env python3
"""
Convert TA dataset to nnUNet format for medical image segmentation training.

TA dataset structure:
- TA/Healthy/Images/*.png
- TA/Healthy/Masks/*.png
- TA/Pathological/Images/*.png
- TA/Pathological/Masks/*.png

nnUNet format:
- data/nnUNet_raw/Dataset803_TA/
  - imagesTr/{case_id}_0000.png
  - labelsTr/{case_id}.png
  - dataset.json
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

def setup_dataset_structure(dataset_id=803, dataset_name="TA"):
    """Setup nnUNet dataset directory structure."""
    base_dir = Path("data/nnUNet_raw")
    dataset_dir = base_dir / f"Dataset{dataset_id:03d}_{dataset_name}"

    # Create directories
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_ts_dir = dataset_dir / "imagesTs"  # Test images (optional)

    for dir_path in [images_dir, labels_dir, images_ts_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    return dataset_dir, images_dir, labels_dir

def convert_images_and_labels(source_dir, images_dir, labels_dir, case_prefix):
    """Convert images and labels from source to nnUNet format."""
    images_path = Path(source_dir) / "Images"
    masks_path = Path(source_dir) / "Masks"

    image_files = sorted(images_path.glob("*.png"))
    print(f"Found {len(image_files)} images in {images_path}")

    case_counter = 0
    converted_cases = []

    for img_file in image_files:
        # Find corresponding mask
        mask_file = masks_path / img_file.name

        if not mask_file.exists():
            print(f"Warning: No mask found for {img_file.name}")
            continue

        # Generate case ID
        case_id = f"{case_prefix}{case_counter:04d}"
        case_counter += 1

        # Copy image with nnUNet naming convention
        target_img = images_dir / f"{case_id}_0000.png"
        target_mask = labels_dir / f"{case_id}.png"

        # Copy files
        shutil.copy2(img_file, target_img)
        shutil.copy2(mask_file, target_mask)

        converted_cases.append(case_id)

        if len(converted_cases) % 500 == 0:
            print(f"Processed {len(converted_cases)} cases...")

    print(f"Converted {len(converted_cases)} cases from {source_dir}")
    return converted_cases

def create_dataset_json(dataset_dir, all_cases, dataset_id=803):
    """Create dataset.json file required by nnUNet."""

    # Calculate train/test split (80/20)
    n_train = int(0.8 * len(all_cases))
    train_cases = all_cases[:n_train]
    test_cases = all_cases[n_train:]

    dataset_json = {
        "channel_names": {
            "0": "Image"
        },
        "labels": {
            "background": 0,
            "foreground": 1
        },
        "numTraining": len(train_cases),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "SimpleITKIO",
        "dataset_name": "TA_Segmentation",
        "description": "TA dataset for medical image segmentation - Healthy vs Pathological",
        "reference": "Medical Image Segmentation Dataset",
        "licence": "Research Use",
        "numTest": len(test_cases),
        "training": [{"image": f"./imagesTr/{case}_0000.png", "label": f"./labelsTr/{case}.png"} for case in train_cases],
        "test": [f"./imagesTs/{case}_0000.png" for case in test_cases]
    }

    # Save dataset.json
    json_path = dataset_dir / "dataset.json"
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Created dataset.json with {len(train_cases)} training and {len(test_cases)} test cases")
    return train_cases, test_cases

def verify_conversion(dataset_dir, sample_size=5):
    """Verify the conversion by checking a few samples."""
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"

    image_files = sorted(images_dir.glob("*.png"))

    print(f"\\nVerification Results:")
    print(f"Total converted images: {len(image_files)}")

    # Check a few samples
    for i, img_file in enumerate(image_files[:sample_size]):
        case_id = img_file.stem.replace('_0000', '')
        label_file = labels_dir / f"{case_id}.png"

        if label_file.exists():
            # Load and check dimensions
            img = Image.open(img_file)
            label = Image.open(label_file)
            print(f"  {case_id}: Image {img.size}, Label {label.size} - {'✓' if img.size == label.size else '✗'}")
        else:
            print(f"  {case_id}: Missing label file - ✗")

def main():
    """Main conversion function."""
    print("🔄 Converting TA dataset to nnUNet format...")

    # Setup dataset structure
    dataset_dir, images_dir, labels_dir = setup_dataset_structure()
    print(f"Created dataset directory: {dataset_dir}")

    all_cases = []

    # Convert healthy images
    print("\\n📁 Converting Healthy images...")
    healthy_cases = convert_images_and_labels("TA/Healthy", images_dir, labels_dir, "healthy_")
    all_cases.extend(healthy_cases)

    # Convert pathological images
    print("\\n📁 Converting Pathological images...")
    pathological_cases = convert_images_and_labels("TA/Pathological", images_dir, labels_dir, "patho_")
    all_cases.extend(pathological_cases)

    # Create dataset.json
    print("\\n📄 Creating dataset.json...")
    train_cases, test_cases = create_dataset_json(dataset_dir, all_cases)

    # Verify conversion
    verify_conversion(dataset_dir)

    print(f"\\n✅ TA dataset conversion completed!")
    print(f"Dataset location: {dataset_dir}")
    print(f"Total cases: {len(all_cases)}")
    print(f"Training cases: {len(train_cases)}")
    print(f"Test cases: {len(test_cases)}")
    print(f"\\n🎯 Next steps:")
    print(f"1. Set environment variable: export nnUNet_raw={Path('data/nnUNet_raw').absolute()}")
    print(f"2. Run preprocessing: nnUNetv2_plan_and_preprocess -d 803 --verify_dataset_integrity")
    print(f"3. Start training: nnUNetv2_train 803 2d all -tr nnUNetTrainerLightMUNet")

if __name__ == "__main__":
    main()