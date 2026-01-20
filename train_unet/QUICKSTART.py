#!/usr/bin/env python3
"""
Quick Start Guide - U-Net Dataset Preparation
Run this script to see a complete workflow example
"""

import sys
from pathlib import Path

def print_quick_start():
    """Print quick start guide for the dataset preparation workflow."""
    
    print("\n" + "="*70)
    print(" "*15 + "U-NET DATASET PREPARATION - QUICK START")
    print("="*70)
    
    print("\n📋 WORKFLOW OVERVIEW")
    print("-" * 70)
    print("""
1. Capture images → Use ROS 2 image_capture_node
2. Annotate → Use LabelMe to create JSON annotations
3. Prepare dataset → Run prepare_dataset.py
4. Visualize → Use train_unet_notebook.ipynb
5. Train U-Net → (Next step - coming soon)
    """)
    
    print("="*70)
    print("STEP 1: IMAGE CAPTURE (ROS 2)")
    print("="*70)
    print("""
# Terminal 1: Launch simulation
ros2 launch your_simulation_package your_world.launch.py

# Terminal 2: Capture images
cd /path/to/qcar2_LaneSeg-ACC
source /path/to/ros2_ws/install/setup.bash
ros2 run qcar2_laneseg_acc image_capture_node

# Images saved to: train_unet/training_data/raw_images/
# Resolution: 640x480 (auto-detected)
# Throttle: 1.0 seconds between captures
    """)
    
    print("="*70)
    print("STEP 2: ANNOTATION (LabelMe)")
    print("="*70)
    print("""
# Install LabelMe
pip install labelme

# Open LabelMe
cd train_unet
labelme training_data/raw_images

# Label Guidelines:
  - Clase 0: 'fondo', 'vereda', 'obstaculo' → Negro
  - Clase 1: 'camino', 'asfalto', 'road' → Azul
  - Clase 2: 'linea', 'lineas', 'lane' → Amarillo
  - Clase 3: 'borde', 'bordes', 'edge' → Rojo

# Save each annotation as .json in the same directory
    """)
    
    print("="*70)
    print("STEP 3: PREPARE DATASET")
    print("="*70)
    print("""
cd train_unet

# Basic usage (80/20 train/val split)
python prepare_dataset.py \\
  --input training_data/raw_images \\
  --output training_data \\
  --val-split 0.2

# Advanced usage (70/20/10 train/val/test split)
python prepare_dataset.py \\
  --input training_data/raw_images \\
  --output training_data \\
  --val-split 0.2 \\
  --test-split 0.1 \\
  --seed 42

# Output structure:
# training_data/
#   └── dataset_images/
#       ├── train/
#       │   ├── images/ (img_00001.png, ...)
#       │   └── masks/  (img_00001.png, ...)
#       ├── val/
#       │   ├── images/
#       │   └── masks/
#       └── test/
#           └── images/
    """)
    
    print("="*70)
    print("STEP 4: VISUALIZE & VALIDATE")
    print("="*70)
    print("""
# Option A: Jupyter Notebook (Recommended)
jupyter notebook train_unet_notebook.ipynb

# Option B: Test script
python test_visualization.py

# Option C: View configuration
python config.py

# The notebook includes:
  - Color legend visualization
  - Random sample display
  - Class distribution analysis
  - Multiple samples grid
  - Overlay visualization
    """)
    
    print("="*70)
    print("STEP 5: PROGRAMMATIC USE (Python Scripts)")
    print("="*70)
    print("""
# In your own Python code:

from prepare_dataset import visualize_mask, overlay_mask_on_image
import config
import cv2

# Load image and mask
image = cv2.imread('dataset_images/train/images/img_00001.png')
mask = cv2.imread('dataset_images/train/masks/img_00001.png', cv2.IMREAD_GRAYSCALE)

# Visualize with colors
colored_mask = visualize_mask(mask, use_colors=True)
cv2.imshow('Colored Mask', colored_mask)

# Overlay on image
overlay = overlay_mask_on_image(image, mask, alpha=0.5)
cv2.imshow('Overlay', overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()
    """)
    
    print("="*70)
    print("📚 ADDITIONAL RESOURCES")
    print("="*70)
    print("""
Files in this directory:
  - config.py              → Class definitions and colors
  - prepare_dataset.py     → Dataset preparation + utilities
  - train_unet_notebook.ipynb → Interactive visualization
  - test_visualization.py  → Quick test script
  - README.md              → Complete documentation
  - CHANGES.md             → Detailed changelog

Color Mapping (Class ID → Color → RGB):
  0 → Negro    → (0, 0, 0)     - Fondo/Vereda/Obstáculos
  1 → Azul     → (0, 0, 255)   - Camino/Asfalto transitable
  2 → Amarillo → (255, 255, 0) - Líneas de tráfico
  3 → Rojo     → (255, 0, 0)   - Bordes de camino
    """)
    
    print("="*70)
    print("⚠️  IMPORTANT NOTES")
    print("="*70)
    print("""
1. Images WITHOUT .json annotations will be IGNORED (with warning)
2. All functions use colors from config.py automatically
3. Masks are single-channel grayscale with values [0, 1, 2, 3]
4. File naming: Images and masks have EXACT same filenames
5. Random seed (default: 42) ensures reproducible splits
    """)
    
    print("="*70)
    print("✅ READY TO START!")
    print("="*70)
    print("""
Current working directory should be: train_unet/

Next steps:
  1. Capture images with ROS 2 image_capture_node
  2. Annotate with LabelMe
  3. Run: python prepare_dataset.py --input training_data/raw_images --output training_data
  4. Open: jupyter notebook train_unet_notebook.ipynb

For help: python prepare_dataset.py --help
    """)
    
    print("="*70 + "\n")


if __name__ == '__main__':
    print_quick_start()
