#!/usr/bin/env python3
"""
Quick test script to verify visualization functions
"""

import numpy as np
import cv2
from pathlib import Path
import config
from prepare_dataset import visualize_mask, overlay_mask_on_image

def test_visualization():
    """Test the visualization functions with a sample mask."""
    print("="*60)
    print("TESTING VISUALIZATION FUNCTIONS")
    print("="*60)
    
    # Create a test mask (640x480)
    height, width = 480, 640
    test_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Simulate segmentation regions
    test_mask[0:120, :] = 0      # Background (top)
    test_mask[120:300, :] = 1    # Road (middle)
    test_mask[300:400, :] = 2    # Lanes (lower middle)
    test_mask[400:480, :] = 3    # Edges (bottom)
    
    print(f"\n✅ Created test mask with shape: {test_mask.shape}")
    print(f"   Unique values: {np.unique(test_mask)}")
    
    # Test colored visualization
    colored_mask = visualize_mask(test_mask, use_colors=True)
    print(f"\n✅ Generated colored mask with shape: {colored_mask.shape}")
    
    # Create a test image
    test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Test overlay
    overlay = overlay_mask_on_image(test_image, test_mask, alpha=0.5)
    print(f"✅ Generated overlay with shape: {overlay.shape}")
    
    # Save test outputs
    output_dir = Path('test_outputs')
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / 'test_mask_colored.png'), colored_mask)
    cv2.imwrite(str(output_dir / 'test_overlay.png'), overlay)
    
    print(f"\n💾 Saved test outputs to: {output_dir}/")
    print(f"   - test_mask_colored.png")
    print(f"   - test_overlay.png")
    
    # Display class colors
    print(f"\n🎨 Color Mapping Verification:")
    for class_id in range(config.NUM_CLASSES):
        name = config.get_class_name(class_id)
        color_bgr = config.get_class_color(class_id, bgr=True)
        print(f"   Clase {class_id}: {name}")
        print(f"     BGR: {color_bgr}")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)

if __name__ == '__main__':
    test_visualization()
