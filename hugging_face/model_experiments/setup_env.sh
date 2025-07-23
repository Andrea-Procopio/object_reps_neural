# Setup script for exp3b_correlation.py virtual environment
# This script creates a virtual environment and installs all dependencies

set -e  # Exit on any error

echo "Setting up virtual environment for exp3b_correlation.py..."

# Check if Python 3.8+ is available
python_version=$(python3 --version 2>&1 | sed 's/Python \([0-9]\+\.[0-9]\+\).*/\1/')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo "Python version: $python_version ✓"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv_exp3b

# Activate virtual environment
echo "Activating virtual environment..."
source venv_exp3b/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (with CUDA support if available)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Test installation
echo "Testing installation..."
python -c "
import torch
import transformers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from PIL import Image
print('✓ All core dependencies installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
"

echo ""
echo "virtual environment setup complete"
echo ""
echo "To activate the environment, run:"
echo "  source venv_exp3b/bin/activate"
echo ""
echo "To run the script, use:"
echo "  python exp3b_correlation.py --help"
echo ""
echo "Example usage:"
echo "  python exp3b_correlation.py \\"
echo "    --images_dir /path/to/exp3b_imgs \\"
echo "    --human_csv /path/to/human_data.csv \\"
echo "    --output_dir /tmp/exp3b_out \\"
echo "    --model_name nvidia/segformer-b0-finetuned-ade-512-512" 