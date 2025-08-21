#!/usr/bin/env python3
"""
setup_build_env.py
Location: /content/drive/MyDrive/Pointcept/Installs/setup_build_env.py

@brief Pointcept build environment setup and dependency installation script
Installation workflow organization and step sequencing assisted by Claude AI (Anthropic).
Multiple prompts used for organizing complex CUDA/PyTorch dependency installation
procedures (abbreviated from extended conversation).

Base Pointcept framework and build tools from:
Pointcept Contributors (2023). Pointcept: A Codebase for Point Cloud Perception Research [online].
[Accessed 2025]. Available from: "https://github.com/Pointcept/Pointcept".
Original Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
"""

#!/usr/bin/env python3


import os
import subprocess
import sys

# Step 1: Build with CUDA 11.8
os.environ['FORCE_CUDA'] = '1'
os.system("pip install ninja")
os.system("python setup.py build_ext --inplace")

# Step 2: Install addict
os.system("pip install addict")

# Step 3: Disable JIT
os.environ['CUMM_DISABLE_JIT'] = '1'
os.environ['SPCONV_DISABLE_JIT'] = '1'

# Step 4: Install plyfile
os.system("pip install plyfile")

# Step 5: Install yapf
os.system("pip install yapf")

# Step 6: Check versions
import torch
import numpy as np
print(torch.__version__)
print(f"NumPy: {np.__version__}")

# Step 7: Override CUDA paths
os.environ["CUDA_HOME"] = "/usr/local/cuda-11.8"
os.environ["PATH"] = "/usr/local/cuda-11.8/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-11.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# Step 8: Verify installation
os.system("ls /usr/local/cuda-11.8/bin/nvcc")
os.system("nvcc --version")
print(f"NumPy: {np.__version__}")

# Step 9: Install sparsehash
os.system("sudo apt-get install -y libsparsehash-dev")

# Step 10: Build pointgroup_ops
os.chdir("/content/drive/MyDrive/Pointcept/libs/pointgroup_ops")
os.system("python setup.py build_ext --inplace")

# Step 11: Main Pointcept installation
def run_command(cmd, check=True):
    """Run shell command"""
    if isinstance(cmd, str):
        cmd = cmd.split()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result

# Check environment
print("============================================================")
print("CHECKING ENVIRONMENT")
print("============================================================")
try:
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ PyTorch Version: {torch.__version__}")
    else:
        print("✗ No GPU detected!")
        sys.exit(1)
except ImportError:
    print("✗ PyTorch not installed!")
    sys.exit(1)

# Get CUDA version
cuda_version = torch.version.cuda.replace(".", "")[:3]
print(f"\n✓ CUDA Version for installations: cu{cuda_version}")

print("\n============================================================")
print("FIXING NUMPY COMPATIBILITY")
print("============================================================")

# Apply numpy fixes early
import numpy as np
for attr in ['bool', 'int', 'float', 'complex', 'object', 'str']:
    if not hasattr(np, attr):
        setattr(np, attr, getattr(__builtins__, attr, type))
print(f"✓ NumPy {np.__version__} compatibility fixes applied")

print("\n============================================================")
print("INSTALLING DEPENDENCIES")
print("============================================================")

# Update pip
run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

# Install all dependencies at once for speed
deps = [
    "numpy==1.24.0", "scikit-learn==1.3.2",
    "h5py", "pyyaml", "sharedarray", "tensorboard", "tensorboardx",
    "yapf", "addict", "einops", "scipy", "plyfile", "termcolor", "timm",
    "opencv-python", "open3d", "trimesh", "rtree"
]
print("Installing basic dependencies...")
run_command([sys.executable, "-m", "pip", "install"] + deps)

# Install PyG dependencies
print("\nInstalling PyTorch Geometric dependencies...")
torch_version = torch.__version__.split('+')[0]
pyg_url = f"https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_version}.html"

for dep in ["torch-scatter", "torch-sparse", "torch-cluster", "torch-geometric"]:
    print(f"Installing {dep}...")
    run_command([sys.executable, "-m", "pip", "install", dep, "-f", pyg_url])

# Install spconv
print("\nInstalling spconv...")
run_command([sys.executable, "-m", "pip", "install", f"spconv-cu{cuda_version}"])

print("\n============================================================")
print("CLONING POINTCEPT")
print("============================================================")

# Clone Pointcept
if not os.path.exists('/content/Pointcept'):
    run_command(["git", "clone", "https://github.com/Pointcept/Pointcept.git", "/content/Pointcept"])
    print("✓ Cloned Pointcept")
else:
    print("✓ Pointcept already exists")

# Detect GPU architecture
gpu_name = torch.cuda.get_device_name(0)
gpu_arch_map = {
    "T4": "7.5",
    "V100": "7.0",
    "A100": "8.0",
    "L4": "8.9"
}

gpu_arch = "8.0"  # Default for A100
for key, arch in gpu_arch_map.items():
    if key in gpu_name:
        gpu_arch = arch
        break

print(f"✓ GPU: {gpu_name}, Using CUDA architecture: {gpu_arch}")
os.environ['TORCH_CUDA_ARCH_LIST'] = gpu_arch

print("\n============================================================")
print("BUILDING POINTOPS")
print("============================================================")

# Build pointops using pip (more reliable in Colab)
print("Building pointops with pip...")
result = run_command([
    sys.executable, "-m", "pip", "install",
    "/content/Pointcept/libs/pointops", "-v"
], check=False)

if result.returncode != 0:
    print("First attempt failed, trying with --no-build-isolation...")
    run_command([
        sys.executable, "-m", "pip", "install",
        "/content/Pointcept/libs/pointops",
        "--no-build-isolation", "-v"
    ])

# Test pointops
try:
    import pointops
    print("✓ pointops imported successfully")
except ImportError:
    print("✗ pointops import failed, trying alternative fix...")
    # Try building directly
    original_dir = os.getcwd()
    os.chdir('/content/Pointcept/libs/pointops')
    run_command([sys.executable, "setup.py", "build_ext", "--inplace"])
    run_command([sys.executable, "setup.py", "install"])
    os.chdir(original_dir)

print("\n============================================================")
print("BUILDING POINTGROUP_OPS")
print("============================================================")

# Build pointgroup_ops
print("Building pointgroup_ops with pip...")
result = run_command([
    sys.executable, "-m", "pip", "install",
    "/content/Pointcept/libs/pointgroup_ops", "-v"
], check=False)

if result.returncode != 0:
    print("First attempt failed, trying with --no-build-isolation...")
    run_command([
        sys.executable, "-m", "pip", "install",
        "/content/Pointcept/libs/pointgroup_ops",
        "--no-build-isolation", "-v"
    ])

# Test pointgroup_ops
try:
    import pointgroup_ops
    print("✓ pointgroup_ops imported successfully")
except ImportError:
    print("✗ pointgroup_ops import failed, trying alternative fix...")
    original_dir = os.getcwd()
    os.chdir('/content/Pointcept/libs/pointgroup_ops')
    run_command([sys.executable, "setup.py", "build_ext", "--inplace"])
    run_command([sys.executable, "setup.py", "install"])
    os.chdir(original_dir)

print("\n============================================================")
print("FINAL VERIFICATION")
print("============================================================")

# Add paths
sys.path.insert(0, '/content/Pointcept')
sys.path.insert(0, '/content/Pointcept/libs/pointops')
sys.path.insert(0, '/content/Pointcept/libs/pointgroup_ops')

# Test all imports
modules = {
    "torch": lambda m: m.__version__,
    "numpy": lambda m: m.__version__,
    "sklearn": lambda m: m.__version__,
    "torch_cluster": lambda m: "installed",
    "pointops": lambda m: "installed",
    "pointgroup_ops": lambda m: "installed",
    "spconv": lambda m: m.__version__
}

all_success = True
for module_name, version_func in modules.items():
    try:
        if module_name == "sklearn":
            import sklearn
            module = sklearn
        else:
            module = __import__(module_name)
        version = version_func(module)
        print(f"✓ {module_name} ({version})")
    except ImportError as e:
        print(f"✗ {module_name}: {str(e)}")
        all_success = False

if all_success:
    print("\n✅ Installation completed successfully!")
    print("\nTo use Pointcept, add these lines at the start of your code:")
    print("```python")
    print("import sys")
    print("sys.path.insert(0, '/content/Pointcept')")
    print("sys.path.insert(0, '/content/Pointcept/libs/pointops')")
    print("sys.path.insert(0, '/content/Pointcept/libs/pointgroup_ops')")
    print("```")
else:
    print("\n⚠️  Some modules failed to import.")
    print("Try restarting the runtime and running this script again.")

# Quick functionality test
if all_success:
    print("\n============================================================")
    print("QUICK FUNCTIONALITY TEST")
    print("============================================================")
    try:
        import torch
        import pointops
        import pointgroup_ops

        # Test basic operations
        coords = torch.rand(100, 3).cuda()
        print("✓ Created test tensor on GPU")
        print("✓ Basic functionality test passed!")
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")

# Step 12: Import pointgroup_ops and pointops
import pointgroup_ops
import pointops

# Step 13: Install sparsehash again
os.system("apt-get install -y libsparsehash-dev")

# Step 14: Change to Pointcept directory
os.chdir("/content/drive/MyDrive/Pointcept")

# Step 15: Set library paths
import torch
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ['PYTHONPATH'] = ':'.join(sys.path)

# Step 16: Import and add to path
import pointgroup_ops
import pointops
sys.path.insert(0, '/content/drive/MyDrive/Pointcept/libs')

# Step 17: Install tensorboardX
os.system("pip install tensorboardX")

# Step 18: Test imports
sys.path.insert(0, '/content/drive/MyDrive/Pointcept/libs')
from pointops._C import knn_query_cuda
print("pointops._C imported successfully!")

import torch_scatter
print("torch-scatter imported successfully!")

# Step 19: Reinstall numpy
os.system("pip uninstall -y numpy")
os.system("pip uninstall -y numpy")
os.system("rm -rf /usr/local/lib/python*/site-packages/numpy*")
os.system("rm -rf /usr/local/lib/python*/dist-packages/numpy*")
os.system("pip install numpy==1.24.3 --no-cache-dir")

import numpy as np
print(f"Numpy: {np.__version__}")
print(f"Has np.bool: {hasattr(np, 'bool')}")

import torch_cluster
print("✅ torch_cluster imported!")

# Step 20: Apply numpy patches
import numpy as np
np.bool = np.bool_
np.int = int
np.float = float
np.complex = complex
np.object = object
np.str = str

print(f"NumPy version: {np.__version__}")
print(f"Patch applied - np.bool is now: {np.bool}")

import torch_cluster

# Step 21: Change to Pointcept
os.chdir("/content/drive/MyDrive/Pointcept")

# Step 22: Cache settings
empty_cache = True
empty_cache_per_epoch = True

import torch
torch.cuda.empty_cache()
import gc
gc.collect()

# Step 23: Final test
from pointgroup_ops import ballquery_batch_p, bfs_cluster

print("Ops loaded:", ballquery_batch_p, bfs_cluster)
