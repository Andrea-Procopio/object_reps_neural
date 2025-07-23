# Setup Guide for exp3b_correlation.py

This guide will help you set up a virtual environment to run the `exp3b_correlation.py` script and related experiments.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Quick Setup (Recommended)

1. **Navigate to the experiment directory:**
   ```bash
   cd hugging_face/model_experiments
   ```

2. **Run the automated setup script:**
   ```bash
   ./setup_env.sh
   ```

   This script will:
   - Check Python version compatibility
   - Create a virtual environment named `venv_exp3b`
   - Install PyTorch with CUDA support (if available)
   - Install all required dependencies
   - Test the installation

3. **Activate the virtual environment:**
   ```bash
   source venv_exp3b/bin/activate
   ```

## Manual Setup

If you prefer to set up manually or the automated script doesn't work:

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv_exp3b
   ```

2. **Activate the environment:**
   ```bash
   source venv_exp3b/bin/activate
   ```

3. **Upgrade pip:**
   ```bash
   pip install --upgrade pip
   ```

4. **Install PyTorch:**
   ```bash
   # For CUDA support (if you have an NVIDIA GPU):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU only:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

5. **Install other dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Testing the Installation

After setup, test that everything works:

```bash
# Activate the environment
source venv_exp3b/bin/activate

# Test imports
python -c "
import torch
import transformers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from PIL import Image
print('✓ All dependencies installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

## Running the Script

Once the environment is set up:

1. **Activate the environment:**
   ```bash
   source venv_exp3b/bin/activate
   ```

2. **Check the script help:**
   ```bash
   python exp3b_correlation.py --help
   ```

3. **Run the experiment:**
   ```bash
   python exp3b_correlation.py \
     --images_dir /path/to/exp3b_imgs \
     --human_csv /path/to/human_data.csv \
     --output_dir /tmp/exp3b_out \
     --model_name nvidia/segformer-b3-finetuned-ade-512-512
   ```

## Troubleshooting

### Common Issues

1. **Python version too old:**
   - Ensure you have Python 3.8+ installed
   - Use `python3 --version` to check

2. **CUDA not available:**
   - The script will work with CPU, but will be slower
   - Install CUDA drivers if you have an NVIDIA GPU

3. **Import errors:**
   - Make sure the virtual environment is activated
   - Try reinstalling dependencies: `pip install -r requirements.txt --force-reinstall`

4. **Permission denied on setup script:**
   - Make it executable: `chmod +x setup_env.sh`

### Getting Help

If you encounter issues:

1. Check that all dependencies are installed correctly
2. Ensure the virtual environment is activated
3. Verify Python version compatibility
4. Check that you have sufficient disk space for model downloads

## Dependencies Overview

The main dependencies include:

- **PyTorch & TorchVision**: Deep learning framework
- **Transformers**: Hugging Face model library
- **NumPy & Pandas**: Data processing
- **Matplotlib**: Plotting and visualization
- **SciPy**: Scientific computing
- **Pillow**: Image processing
- **scikit-image**: Image analysis

## Next Steps

After successful setup, you can:

1. Run the correlation experiment
2. Explore other experiments in the directory
3. Modify parameters and thresholds
4. Analyze results in the output directory 