#!/usr/bin/env python3
"""
Dataset Preparation Script for U-Net Semantic Segmentation
Author: hackbrian
Description: Converts LabelMe annotations to organized PyTorch dataset structure
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Import configuration from centralized config module
import config



def parse_labelme_json(json_path: str) -> Tuple[List[np.ndarray], List[int], Tuple[int, int]]:
    """
    Parse LabelMe JSON file and extract polygon coordinates and class labels.
    
    Args:
        json_path: Path to LabelMe JSON file
        
    Returns:
        Tuple of (polygon_list, class_list, image_shape)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_height = data['imageHeight']
    image_width = data['imageWidth']
    image_shape = (image_height, image_width)
    
    polygons = []
    classes = []
    
    for shape in data.get('shapes', []):
        label = shape['label'].lower().strip()
        points = np.array(shape['points'], dtype=np.int32)
        
        # Map label to class ID using config module
        class_id = config.get_class_id(label, default=0)
        
        polygons.append(points)
        classes.append(class_id)
    
    return polygons, classes, image_shape


def create_mask_from_polygons(polygons: List[np.ndarray], classes: List[int], 
                               image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a segmentation mask from polygon coordinates.
    
    Args:
        polygons: List of polygon coordinates
        classes: List of class IDs for each polygon
        image_shape: (height, width) of the output mask
        
    Returns:
        Grayscale mask with pixel values [0, 1, 2, 3]
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    # Draw polygons in order (later polygons may overlap earlier ones)
    for polygon, class_id in zip(polygons, classes):
        cv2.fillPoly(mask, [polygon], color=class_id)
    
    return mask


def organize_dataset(input_dir: str, output_dir: str, val_split: float = 0.2, 
                     test_split: float = 0.0, random_seed: int = 42):
    """
    Main function to organize LabelMe data into training dataset structure.
    
    Args:
        input_dir: Directory containing raw images and JSON files
        output_dir: Output directory for organized dataset
        val_split: Fraction of data for validation (default: 0.2)
        test_split: Fraction of data for test set (default: 0.0)
        random_seed: Random seed for reproducibility
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all JSON files
    json_files = list(input_path.glob('*.json'))
    
    if len(json_files) == 0:
        print(f"❌ No JSON files found in {input_dir}")
        return
    
    print(f"📁 Found {len(json_files)} JSON annotation files")
    
    # Prepare file pairs (image, json)
    valid_pairs = []
    
    for json_file in json_files:
        # Try to find corresponding image (png, jpg, jpeg)
        image_name = json_file.stem
        image_file = None
        
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            candidate = json_file.parent / f"{image_name}{ext}"
            if candidate.exists():
                image_file = candidate
                break
        
        if image_file:
            valid_pairs.append((image_file, json_file))
        else:
            print(f"⚠️  Warning: No image found for {json_file.name}")
    
    print(f"✅ Found {len(valid_pairs)} valid image-annotation pairs")
    
    if len(valid_pairs) == 0:
        print("❌ No valid pairs found. Exiting.")
        return
    
    # Split data
    if test_split > 0:
        train_val_pairs, test_pairs = train_test_split(
            valid_pairs, test_size=test_split, random_state=random_seed
        )
    else:
        train_val_pairs = valid_pairs
        test_pairs = []
    
    if val_split > 0:
        train_pairs, val_pairs = train_test_split(
            train_val_pairs, test_size=val_split, random_state=random_seed
        )
    else:
        train_pairs = train_val_pairs
        val_pairs = []
    
    print(f"\n📊 Dataset Split:")
    print(f"   Train: {len(train_pairs)} samples")
    print(f"   Val:   {len(val_pairs)} samples")
    print(f"   Test:  {len(test_pairs)} samples")
    
    # Create directory structure
    # New structure: output_dir/dataset_images/train, val, test
    dataset_base = output_path / 'dataset_images'
    
    directories = {
        'train_images': dataset_base / 'train' / 'images',
        'train_masks': dataset_base / 'train' / 'masks',
        'val_images': dataset_base / 'val' / 'images',
        'val_masks': dataset_base / 'val' / 'masks',
        'test_images': dataset_base / 'test' / 'images',
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Created directory structure in: {dataset_base}")
    
    # Process and copy files
    def process_split(pairs, split_name, save_masks=True):
        print(f"\n🔄 Processing {split_name} set...")
        
        for idx, (image_file, json_file) in enumerate(pairs, start=1):
            # Generate sequential filename
            new_name = f"img_{idx:05d}.png"
            
            # Copy image
            image_dest = directories[f'{split_name}_images'] / new_name
            shutil.copy2(image_file, image_dest)
            
            # Generate and save mask
            if save_masks:
                try:
                    polygons, classes, image_shape = parse_labelme_json(str(json_file))
                    mask = create_mask_from_polygons(polygons, classes, image_shape)
                    
                    mask_dest = directories[f'{split_name}_masks'] / new_name
                    cv2.imwrite(str(mask_dest), mask)
                    
                    # Verify unique values
                    unique_vals = np.unique(mask)
                    if not all(v in range(config.NUM_CLASSES) for v in unique_vals):
                        print(f"   ⚠️  Warning: {new_name} has unexpected class values: {unique_vals}")
                    
                except Exception as e:
                    print(f"   ❌ Error processing {json_file.name}: {e}")
                    continue
            
            if (idx % 10 == 0) or (idx == len(pairs)):
                print(f"   Processed {idx}/{len(pairs)} files...")
    
    # Process all splits
    if train_pairs:
        process_split(train_pairs, 'train', save_masks=True)
    
    if val_pairs:
        process_split(val_pairs, 'val', save_masks=True)
    
    if test_pairs:
        process_split(test_pairs, 'test', save_masks=False)
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"\n📍 Output structure:")
    print(f"   {output_path}/")
    print(f"   └── dataset_images/")
    print(f"       ├── train/")
    print(f"       │   ├── images/ ({len(train_pairs)} files)")
    print(f"       │   └── masks/  ({len(train_pairs)} files)")
    print(f"       ├── val/")
    print(f"       │   ├── images/ ({len(val_pairs)} files)")
    print(f"       │   └── masks/  ({len(val_pairs)} files)")
    if test_pairs:
        print(f"       └── test/")
        print(f"           └── images/ ({len(test_pairs)} files)")
    
    print(f"\n💡 Class Mapping Used:")
    for class_id in range(config.NUM_CLASSES):
        class_name = config.get_class_name(class_id)
        color = config.get_class_color(class_id)
        print(f"   Clase {class_id}: {class_name} (RGB: {color})")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare LabelMe dataset for U-Net semantic segmentation training'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing raw images and JSON annotations'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for organized dataset'
    )
    
    parser.add_argument(
        '--val-split',
        type=float,
        default=config.DEFAULT_VAL_SPLIT,
        help='Validation split ratio (default: 0.2)'
    )
    
    parser.add_argument(
        '--test-split',
        type=float,
        default=config.DEFAULT_TEST_SPLIT,
        help='Test split ratio (default: 0.0)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=config.DEFAULT_RANDOM_SEED,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("DATASET PREPARATION FOR U-NET TRAINING")
    print("="*60)
    
    organize_dataset(
        input_dir=args.input,
        output_dir=args.output,
        val_split=args.val_split,
        test_split=args.test_split,
        random_seed=args.seed
    )


if __name__ == '__main__':
    main()
