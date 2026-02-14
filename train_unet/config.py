"""
Configuration Module for U-Net Semantic Segmentation
Author: hackbrian
Description: Centralized configuration for class mapping, colors, and dataset parameters
"""

# ============================================================================
# CLASS MAPPING CONFIGURATION
# ============================================================================

# Semantic segmentation class definitions
# Each class has a unique integer ID used in the mask
CLASS_NAMES = {
    0: "Fondo / Vereda / Obstáculos",
    1: "Camino / Asfalto transitable",
    2: "Líneas de tráfico",
    3: "Bordes de camino"
}

# Label to class ID mapping for LabelMe annotations
# Multiple label variations map to the same class ID
LABEL_TO_CLASS = {
    # Clase 0: Background / Sidewalk / Obstacles
    'fondo': 0,
    'background': 0,
    'vereda': 0,
    'sidewalk': 0,
    'obstaculo': 0,
    'obstaculos': 0,
    'obstacle': 0,
    'obstacles': 0,
    
    # Clase 1: Road / Drivable asphalt
    'camino': 1,
    'asfalto': 1,
    'road': 1,
    'drivable': 1,
    
    # Clase 2: Traffic lanes / Lane markings
    'linea': 2,
    'lineas': 2,
    'lane': 2,
    'lanes': 2,
    'marking': 2,
    'markings': 2,
    
    # Clase 3: Road edges
    'borde': 3,
    'bordes': 3,
    'edge': 3,
    'edges': 3,
}

# Number of classes (for U-Net output layer)
NUM_CLASSES = 4

# ============================================================================
# COLOR MAPPING FOR VISUALIZATION
# ============================================================================

# RGB colors for each class (for visualization purposes)
CLASS_COLORS = {
    0: (0, 0, 0),         # Black - Background
    1: (0, 0, 255),       # Blue - Road
    2: (255, 255, 0),     # Yellow - Lanes
    3: (255, 0, 0),       # Red - Edges
}

# BGR colors (OpenCV format)
CLASS_COLORS_BGR = {
    0: (0, 0, 0),         # Black - Background
    1: (255, 0, 0),       # Blue - Road
    2: (0, 255, 255),     # Yellow - Lanes
    3: (0, 0, 255),       # Red - Edges
}

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Default split ratios
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_TEST_SPLIT = 0.0
DEFAULT_RANDOM_SEED = 42

# Image properties (detected from camera: 640x480)
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480

# ============================================================================
# TRAINING CONFIGURATION (for future use)
# ============================================================================

# U-Net input size (should match camera resolution or be adjusted accordingly)
UNET_INPUT_WIDTH = 640
UNET_INPUT_HEIGHT = 480

# Data augmentation parameters (for future training script)
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'vertical_flip': False,
    'rotation_range': 5,  # degrees
    'brightness_range': 0.2,
    'zoom_range': 0.1,
}

# Training hyperparameters (for future training script)
TRAINING_CONFIG = {
    'batch_size': 8,
    'learning_rate': 1e-4,
    'num_epochs': 100,
    'early_stopping_patience': 15,
    'optimizer': 'adam',
    'loss_function': 'categorical_crossentropy',
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_class_id(label: str, default: int = 0) -> int:
    """
    Get class ID for a given label string.
    
    Args:
        label: Label string from annotation
        default: Default class ID if label not found
        
    Returns:
        Class ID (0-3)
    """
    label_lower = label.lower().strip()
    return LABEL_TO_CLASS.get(label_lower, default)


def get_class_name(class_id: int) -> str:
    """
    Get class name for a given class ID.
    
    Args:
        class_id: Class ID (0-3)
        
    Returns:
        Class name string
    """
    return CLASS_NAMES.get(class_id, "Unknown")


def get_class_color(class_id: int, bgr: bool = False) -> tuple:
    """
    Get color for a given class ID.
    
    Args:
        class_id: Class ID (0-3)
        bgr: If True, return BGR format (OpenCV), else RGB
        
    Returns:
        Color tuple (R, G, B) or (B, G, R)
    """
    if bgr:
        return CLASS_COLORS_BGR.get(class_id, (0, 0, 0))
    else:
        return CLASS_COLORS.get(class_id, (0, 0, 0))


def print_class_info():
    """Print class mapping information."""
    print("\n" + "="*60)
    print("SEMANTIC SEGMENTATION CLASS CONFIGURATION")
    print("="*60)
    print(f"Number of classes: {NUM_CLASSES}\n")
    
    for class_id in range(NUM_CLASSES):
        color = get_class_color(class_id)
        name = get_class_name(class_id)
        print(f"  Clase {class_id}: {name}")
        print(f"    RGB Color: {color}")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    # Display configuration when run directly
    print_class_info()
    
    print("Label Mappings:")
    for label, class_id in sorted(LABEL_TO_CLASS.items()):
        print(f"  '{label}' → Class {class_id} ({get_class_name(class_id)})")
