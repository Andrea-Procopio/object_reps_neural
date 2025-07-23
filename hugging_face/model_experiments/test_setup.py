#!/usr/bin/env python3
"""
Test script to verify that the virtual environment setup is working correctly.
Run this after setting up the environment to ensure everything is ready.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import scipy.stats
        print(f"✓ SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"✗ SciPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        import PIL
        print(f"✓ Pillow {PIL.__version__}")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    try:
        import skimage
        print(f"✓ scikit-image {skimage.__version__}")
    except ImportError as e:
        print(f"✗ scikit-image import failed: {e}")
        return False
    
    return True

def test_project_imports():
    """Test that project-specific modules can be imported."""
    print("\nTesting project imports...")
    
    # Add current directory to path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    try:
        from segformer.segformer_interface import SegFormerInterface
        print("✓ SegFormerInterface imported successfully")
    except ImportError as e:
        print(f"✗ SegFormerInterface import failed: {e}")
        return False
    
    try:
        from exp3Change import ChangeDetectionExperiment
        print("✓ ChangeDetectionExperiment imported successfully")
    except ImportError as e:
        print(f"✗ ChangeDetectionExperiment import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test that a SegFormer model can be loaded."""
    print("\nTesting model loading...")
    
    try:
        from segformer.segformer_interface import SegFormerInterface
        
        # Create interface with a small model for testing
        model_if = SegFormerInterface(
            model_name="nvidia/segformer-b0-finetuned-ade-512-512"
        )
        print("✓ SegFormerInterface created successfully")
        
        # Test model loading (this will download the model)
        print("  Loading model (this may take a moment)...")
        model_if.load_model()
        print("✓ Model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing exp3b_correlation.py environment setup...\n")
    
    # Test basic imports
    if not test_imports():
        print("\n❌ Basic imports failed. Check your virtual environment setup.")
        return False
    
    # Test project imports
    if not test_project_imports():
        print("\n❌ Project imports failed. Check the import paths.")
        return False
    
    # Test model loading (optional - can be slow)
    print("\nTesting model loading (optional - may take time to download)...")
    if not test_model_loading():
        print("⚠️  Model loading failed, but this might be due to network issues.")
        print("   The script should still work for basic functionality.")
    
    print("\n🎉 Environment setup test completed successfully!")
    print("\nYou can now run:")
    print("  python exp3b_correlation.py --help")
    print("\nExample usage:")
    print("  python exp3b_correlation.py \\")
    print("    --images_dir /path/to/exp3b_imgs \\")
    print("    --human_csv /path/to/human_data.csv \\")
    print("    --output_dir /tmp/exp3b_out \\")
    print("    --model_name nvidia/segformer-b3-finetuned-ade-512-512")
    
    return True

if __name__ == "__main__":
    main() 