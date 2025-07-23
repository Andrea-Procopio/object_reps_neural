# Experiment 3B - Model-Human Correlation Analysis

## Overview

This script performs correlation analysis between model predictions and human judgments for object-change detection tasks. It compares how well a computer vision model (specifically SegFormer) can detect changes in objects compared to human observers.

## Purpose

The script evaluates the correlation between:
- **Model decisions**: Binary classifications (same/different) based on area change ratios
- **Human judgments**: Mean responses from human participants on the same image pairs

## How It Works

1. **Image Processing**: Takes BEFORE/AFTER image pairs (with suffixes `_init` and `_out`)
2. **Segmentation**: Runs the ChangeDetectionExperiment to generate segmentation masks
3. **Area Calculation**: Computes `|area_after - area_before| / area_before` for each pair
4. **Threshold Sweep**: Tests multiple thresholds (0.02 to 0.98) to classify pairs as "different" (1) or "same" (0)
5. **Correlation Analysis**: Calculates Pearson and Spearman correlations between model and human decisions
6. **Results**: Saves comprehensive results including JSON data, summary text, and visualization plots

## Prerequisites

### Virtual Environment
```bash
# Activate the virtual environment before running
source venv_exp3b/bin/activate
```

### Dependencies
- Python 3.7+
- torch
- numpy
- pandas
- matplotlib
- scipy
- transformers (for SegFormer models)

### Required Files
- `exp3Change.py` - Change detection experiment module
- `segformer/segformer_interface.py` - SegFormer model interface
- Human data CSV file with columns: `shape`, `response`, `fullShapeName`
- Image pairs with naming convention: `basename_init.png` and `basename_out.png`

## Usage

### Basic Usage
```bash
python exp3b_correlation.py \
    --images_dir /path/to/exp3b_imgs \
    --human_csv /path/to/human_data.csv \
    --output_dir /path/to/output \
    --model_name nvidia/segformer-b0-finetuned-ade-512-512 \
    --resume
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--images_dir` | Path | `Exp3b_Images/` | Directory containing image pairs |
| `--human_csv` | Path | `exp3b_data.csv` | CSV file with human judgment data |
| `--output_dir` | Path | `exp3b_results/` | Output directory for results |
| `--model_name` | str | None | Hugging Face model checkpoint name |
| `--thresholds` | str | `0.02,0.04,...,0.98` | Comma-separated threshold values |
| `--resume` | flag | False | Resume from previous run |

### Example with Custom Parameters
```bash
python exp3b_correlation.py \
    --images_dir /Users/andreaprocopio/Desktop/object_reps_neural/hugging_face/model_experiments/Exp3b_Images \
    --human_csv /Users/andreaprocopio/Desktop/object_reps_neural/detr/EXP_3_CHANGE/Data_processed/Data/exp3b_data.csv \
    --output_dir /Users/andreaprocopio/Desktop/object_reps_neural/hugging_face/model_experiments/exp3b_results \
    --model_name nvidia/segformer-b0-finetuned-ade-512-512 \
    --thresholds "0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90" \
    --resume
```

## Input Data Format

### Human Data CSV
The CSV file should contain:
- `shape`: Shape identifier
- `response`: Either "same" or "different"
- `fullShapeName`: Full shape name (used for matching with model results)

Example:
```csv
shape,response,fullShapeName
shape_001,same,shape_001
shape_001,different,shape_001
shape_002,same,shape_002
```

### Image Files
- **Before images**: `basename_init.png`
- **After images**: `basename_out.png`
- Images should be in the same directory specified by `--images_dir`

## Output

The script creates a model-specific subdirectory in the output directory and generates:

### Files Generated
1. **`correlation_results.json`** - Detailed correlation results for all thresholds
2. **`summary.txt`** - Summary with best correlation and threshold
3. **`correlation_vs_threshold.png`** - Plot showing correlation vs threshold

### Output Structure
```
output_dir/
└── model_name/
    ├── correlation_results.json
    ├── summary.txt
    ├── correlation_vs_threshold.png
    └── cde/
        └── [ChangeDetectionExperiment outputs]
```

### JSON Output Format
```json
[
  {
    "threshold": 0.02,
    "pearson": 0.123,
    "spearman": 0.145,
    "n": 150
  },
  ...
]
```

## Key Functions

### `load_human_data(csv_path)`
- Loads and processes human judgment data
- Filters out catch trials
- Returns mean binary responses per shape

### `collect_area_ratios(cde)`
- Extracts area change ratios from ChangeDetectionExperiment results
- Reads from `per_image_detailed.json` files

### `decide_different(area_ratio, thr)`
- Converts area ratio to binary decision based on threshold
- Returns 1 if change exceeds threshold, 0 otherwise

### `correlations(model, human)`
- Calculates Pearson and Spearman correlations
- Handles edge cases (insufficient data, no variance)

## Interpretation

### Correlation Values
- **Pearson r**: Linear correlation coefficient (-1 to 1)
- **Spearman ρ**: Rank correlation coefficient (-1 to 1)
- Higher values indicate better agreement between model and human judgments

### Threshold Selection
- The script identifies the threshold that maximizes Pearson correlation
- This optimal threshold can be used for future model predictions

### Sample Size
- The `n` value indicates how many image pairs were successfully processed
- Larger sample sizes provide more reliable correlation estimates

## Troubleshooting

### Common Issues
1. **No per_image_detailed.json found**: Ensure ChangeDetectionExperiment ran successfully
2. **Missing human data matches**: Check that image base names match `fullShapeName` in CSV
3. **Memory issues**: Consider using smaller model or reducing batch size
4. **CUDA errors**: Ensure GPU memory is sufficient or use CPU-only mode

### Debugging Tips
- Use `--resume` flag to avoid re-running expensive segmentation
- Check console output for correlation values at each threshold
- Verify input file paths and formats
- Monitor GPU memory usage during model inference

## Dependencies and Imports

```python
from __future__ import annotations
import argparse, json, datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import torch
```

## Related Files

- `exp3Change.py` - Main change detection experiment
- `segformer/segformer_interface.py` - SegFormer model wrapper
- Human data CSV files in `detr/EXP_3_CHANGE/Data_processed/Data/`
- Image files in `Exp3b_Images/`

## Notes

- The script automatically creates model-specific output directories to avoid overwriting results
- GPU acceleration is used when available (torch.set_grad_enabled(False))
- The threshold sweep covers a wide range (0.02-0.98) to find optimal performance
- Results are timestamped for reproducibility