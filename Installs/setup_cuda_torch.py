# @brief CUDA and PyTorch installation script with caching for Colab environment
# Installation workflow organization and step sequencing assisted by Claude AI (Anthropic).

import os
import shutil
from pathlib import Path

# Create cache directory
CACHE_DIR = Path('/content/drive/MyDrive/colab_cache')
CACHE_DIR.mkdir(exist_ok=True)

print("="*60)
print("CUDA INSTALLATION")
print("="*60)

# Step 1: Clean existing CUDA
print("\n1. Cleaning existing CUDA installations...")
os.system('apt-get --purge remove "*cuda*" "*nvidia*" -y > /dev/null 2>&1')
os.system('apt-get autoremove -y > /dev/null 2>&1')
os.system('rm -rf /usr/local/cuda*')
print("✓ Cleaned")

# Step 2: Handle CUDA keyring
print("\n2. Installing CUDA keyring...")
keyring_file = "cuda-keyring_1.0-1_all.deb"
cached_keyring = CACHE_DIR / keyring_file

if cached_keyring.exists():
    print("✓ Using cached keyring")
    shutil.copy2(cached_keyring, f'/content/{keyring_file}')
else:
    print("⬇ Downloading keyring...")
    os.system(f'wget -q --show-progress https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/{keyring_file}')
    # Cache for next time
    shutil.copy2(keyring_file, cached_keyring)
    print("✓ Cached for future use")

os.system(f'dpkg -i {keyring_file}')
print("✓ Keyring installed")

# Step 3: Install CUDA toolkit
print("\n3. Installing CUDA Toolkit 11.8...")
os.system('apt-get update -qq')
os.system('apt-get -y install cuda-toolkit-11-8')
os.system('ln -sf /usr/local/cuda-11.8 /usr/local/cuda')
print("✓ CUDA 11.8 installed")

# Step 4: Set environment variables
print("\n4. Setting environment variables...")
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.8'
os.environ['PATH'] = f"/usr/local/cuda-11.8/bin:{os.environ['PATH']}"
os.environ['LD_LIBRARY_PATH'] = f"/usr/local/cuda-11.8/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
print("✓ Environment set")

# Step 5: Verify
print("\n5. Verifying installation...")
os.system('nvcc --version')

print("\n✅ CUDA installation complete!")
print(f"Cached files in: {CACHE_DIR}")

# Set environment (again, exactly as in notebook)
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.8'
os.environ['PATH'] = f"/usr/local/cuda-11.8/bin:{os.environ['PATH']}"
os.environ['LD_LIBRARY_PATH'] = f"/usr/local/cuda-11.8/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Now PyTorch - EXACT order from notebook
print("\n" + "="*60)
print("PYTORCH INSTALLATION")
print("="*60)

# Uninstall first
os.system('pip uninstall -y numpy torch torchvision torchaudio')

# Install with exact versions and order
os.system('pip install numpy==1.24.0 torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --force-reinstall -f https://download.pytorch.org/whl/torch_stable.html')

# Verify versions
import torch
import numpy as np
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

print(f"NumPy: {np.__version__}")


