#!/usr/bin/env python3
"""
Fix label values in TA dataset from 0/255 to 0/1 format for nnUNet compatibility.
nnUNet expects binary labels to have values 0 (background) and 1 (foreground),
but the current masks use 0 and 255.
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def fix_label_values():
    """Convert label values from 0/255 to 0/1."""

    labels_dir = Path("data/nnUNet_raw/Dataset803_TA/labelsTr")

    # Get all label files
    label_files = sorted(labels_dir.glob("*.png"))
    print(f"Found {len(label_files)} label files to convert")

    converted_count = 0

    for label_file in tqdm(label_files, desc="Converting labels"):
        # Load image
        img = Image.open(label_file)
        img_array = np.array(img)

        # Check current unique values
        unique_values = np.unique(img_array)

        # Convert 255 to 1, keep 0 as 0
        if 255 in unique_values:
            # Convert 255 -> 1
            img_array[img_array == 255] = 1

            # Save the corrected image
            corrected_img = Image.fromarray(img_array.astype(np.uint8))
            corrected_img.save(label_file)
            converted_count += 1

    print(f"✅ Converted {converted_count} label files from 0/255 to 0/1 format")

    # Verify a few samples
    print("\n🔍 Verification (first 5 files):")
    for i, label_file in enumerate(label_files[:5]):
        img = Image.open(label_file)
        img_array = np.array(img)
        unique_values = np.unique(img_array)
        print(f"  {label_file.name}: {unique_values}")

if __name__ == "__main__":
    fix_label_values()