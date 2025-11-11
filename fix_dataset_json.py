#!/usr/bin/env python3
"""
Fix dataset.json to match the actual number of training cases.
All 3605 cases are in the training folder, so update the JSON accordingly.
"""

import json
import os
from pathlib import Path

def fix_dataset_json():
    """Update dataset.json to reflect all cases as training data."""

    dataset_dir = Path("data/nnUNet_raw/Dataset803_TA")
    json_file = dataset_dir / "dataset.json"
    images_dir = dataset_dir / "imagesTr"

    # Get all training images
    image_files = sorted(images_dir.glob("*_0000.png"))
    case_ids = [img.stem.replace('_0000', '') for img in image_files]

    print(f"Found {len(case_ids)} actual training cases")

    # Load existing dataset.json
    with open(json_file, 'r') as f:
        dataset_json = json.load(f)

    # Update the configuration
    dataset_json["numTraining"] = len(case_ids)
    dataset_json["numTest"] = 0  # No separate test set
    dataset_json["training"] = [
        {"image": f"./imagesTr/{case}_0000.png", "label": f"./labelsTr/{case}.png"}
        for case in case_ids
    ]
    dataset_json["test"] = []  # Empty test set

    # Save updated dataset.json
    with open(json_file, 'w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"✅ Updated dataset.json:")
    print(f"   Training cases: {dataset_json['numTraining']}")
    print(f"   Test cases: {dataset_json['numTest']}")
    print(f"   Total cases: {len(case_ids)}")

if __name__ == "__main__":
    fix_dataset_json()