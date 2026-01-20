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

## Class Mapping

| Class ID | Label | Color | Description |
|----------|-------|-------|-------------|
| 0 | `fondo`, `vereda`, `obstaculo` | Black | Background / Sidewalk / Obstacles |
| 1 | `camino`, `asfalto`, `road` | Blue | Drivable road / Asphalt |
| 2 | `linea`, `lane` | Yellow | Traffic lane markings |
| 3 | `borde`, `edge` | Red | Road edges |

## Future Extensions

The `config.py` module is designed to be imported by:
- Training scripts
- Inference/prediction scripts
- Visualization tools
- Data augmentation pipelines
