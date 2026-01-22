# Train U-Net Module

This directory contains scripts and utilities for U-Net semantic segmentation training.

## Files

- **`config.py`**: Centralized configuration module
  - Class mapping definitions
  - Color schemes for visualization
  - Dataset parameters
  - Training hyperparameters (for future use)
  - Helper functions

- **`prepare_dataset.py`**: Dataset preparation script
  - Converts LabelMe annotations to PyTorch-ready structure
  - Uses configuration from `config.py`
  - Generates train/val/test splits

- **`training_data/`**: Dataset directory
  - `raw_images/`: Original captures + JSON annotations
  - `train/`: Training set (images + masks)
  - `val/`: Validation set (images + masks)
  - `test/`: Test set (images only)

## Install

  ```bash
  pip install labelme==5.3.1
  ```

### Directory Structure
```
train_unet/
└── training_data
    ├── dataset_images
    │   ├── test
    │   ├── train
    │   └── val
    └── raw_images
```

## Usage

### View Configuration

```bash
python config.py
```

### Prepare Dataset

```bash
python prepare_dataset.py \
  --input training_data/raw_images \
  --output training_data \
  --val-split 0.2
```

**Note**: The script will:
- ✅ Process only images that have corresponding `.json` annotations
- ⚠️ Ignore images without annotations (will show warning)
- 📊 Generate train/val/test splits automatically

### Visualization (Jupyter Notebook)

Use `train_unet_notebook.ipynb` for:
- Interactive dataset exploration
- Random sample visualization
- Class distribution analysis
- Color-coded segmentation overlay

### Programmatic Use

Import functions from `prepare_dataset.py`:

```python
from prepare_dataset import visualize_mask, overlay_mask_on_image
import config

# Visualize mask with colors
colored_mask = visualize_mask(mask, use_colors=True)

# Overlay mask on image
blended = overlay_mask_on_image(image, mask, alpha=0.5)
```

## Class Mapping

| Class ID | Label | Color | Description |
|----------|-------|-------|-------------|
| 0 | `fondo`, `vereda`, `obstaculo` | Black | sidewalk | id 0
| 1 | `camino`, `asfalto`, `road` | Blue | road | id 1
| 2 | `linea`, `lane` | Yellow | lane | id 2
| 3 | `borde`, `edge` | Red | edge | id 3

## Future Extensions

The `config.py` module is designed to be imported by:
- Training scripts
- Inference/prediction scripts
- Visualization tools
- Data augmentation pipelines
