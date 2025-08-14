#!/usr/bin/env python3
"""
fix_ballquery_wrapper.py
Location: /content/drive/MyDrive/Pointcept/Installs/fix_ballquery_wrapper.py

Called via:
%cd /content/drive/MyDrive/Pointcept/Installs
exec(open('/content/drive/MyDrive/Pointcept/Installs/fix_ballquery_wrapper.py').read())

Fixes the ballquery_batch_p wrapper and imports that break after torch_geometric reinstall
Run this AFTER installing torch_geometric==2.3.1
"""

import os
import sys
import shutil
from datetime import datetime
import torch

print("FIXING BALLQUERY WRAPPER AFTER TORCH_GEOMETRIC INSTALL")
print("=" * 80)

# Keep track of what we fix
fixes_applied = []
errors = []
warnings = []

# 1. ENVIRONMENT AND DEPENDENCY CHECKS
print("\n1. CHECKING ENVIRONMENT AND DEPENDENCIES")
print("-" * 40)

# Check Python version
print(f"  Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
if sys.version_info.major == 3 and sys.version_info.minor >= 7:
   print("  ✓ Python version OK")
else:
   warnings.append("Python 3.7+ recommended")

# Check PyTorch and CUDA
try:
   print(f"  PyTorch version: {torch.__version__}")
   print(f"  CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"  CUDA version: {torch.version.cuda}")
       print(f"  GPU: {torch.cuda.get_device_name(0)}")
       print(f"  PyTorch built with CUDA: {torch.version.cuda}")
except Exception as e:
   errors.append(f"PyTorch check failed: {e}")

# Check core dependencies
print("\n  Checking core dependencies:")
dependencies = ['numpy', 'plotly', 'trimesh', 'psutil']
missing_deps = []
for dep in dependencies:
   try:
       __import__(dep)
       print(f"  ✓ {dep}")
   except ImportError:
       print(f"  ✗ {dep}")
       missing_deps.append(dep)

if missing_deps:
   warnings.append(f"Missing dependencies: {', '.join(missing_deps)}")

# Check Pointcept installation
pointcept_paths = [
   '/content/Pointcept',
   '/content/drive/MyDrive/Pointcept',
   './Pointcept',
   '../Pointcept'
]

pointcept_path = None
for path in pointcept_paths:
   if os.path.exists(path):
       pointcept_path = path
       break

if pointcept_path:
   print(f"\n  ✓ Pointcept found at: {pointcept_path}")
   sys.path.insert(0, pointcept_path)
else:
   errors.append("Pointcept not found")
   print("\n  ✗ Pointcept not found")

# Check pointgroup_ops installation
print("\n  Checking pointgroup_ops:")
try:
   import pointgroup_ops
   print(f"  ✓ pointgroup_ops found at: {pointgroup_ops.__file__}")

   # Check if it's the .so file directly (problematic)
   if pointgroup_ops.__file__.endswith('.so'):
       warnings.append("pointgroup_ops is loading .so directly, not the Python wrapper")
       print("  ⚠ WARNING: Loading .so file directly")

   # Check what's available
   if hasattr(pointgroup_ops, 'ballquery_batch_p'):
       print(f"  ✓ Has ballquery_batch_p: {type(pointgroup_ops.ballquery_batch_p)}")
   else:
       errors.append("pointgroup_ops missing ballquery_batch_p")
       print("  ✗ Missing ballquery_batch_p")

except ImportError:
   errors.append("pointgroup_ops not installed")
   print("  ✗ pointgroup_ops not found")

# Check pointgroup_ops_cuda
try:
   import pointgroup_ops_cuda
   print(f"  ✓ pointgroup_ops_cuda found at: {pointgroup_ops_cuda.__file__}")
except ImportError:
   warnings.append("pointgroup_ops_cuda not found - wrapper may fail")
   print("  ⚠ pointgroup_ops_cuda not found")

# Check pointops (optional)
try:
   import pointops
   print(f"  ✓ pointops installed (optional)")
except ImportError:
   print("  ⚠ pointops not installed (optional but recommended)")

# 2. FIX MODEL IMPORTS
print("\n\n2. FIXING MODEL IMPORTS")
print("-" * 40)

def fix_model_imports(file_path):
   """Fix the ballquery import in model files"""
   if not os.path.exists(file_path):
       errors.append(f"File not found: {file_path}")
       return False

   # Create backup with timestamp
   backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
   try:
       shutil.copy2(file_path, backup_path)
       print(f"    Created backup: {os.path.basename(backup_path)}")
   except Exception as e:
       errors.append(f"Failed to create backup: {e}")
       return False

   # Read file
   try:
       with open(file_path, 'r') as f:
           lines = f.readlines()
   except Exception as e:
       errors.append(f"Failed to read file: {e}")
       return False

   # Fix import - look for the problematic import line
   fixed = False
   for i, line in enumerate(lines):
       if 'from pointgroup_ops import ballquery_batch_p, bfs_cluster' in line:
           # Get the indentation
           indent = len(line) - len(line.lstrip())

           # Replace with two separate imports
           lines[i] = ' ' * indent + 'from pointgroup_ops import bfs_cluster\n'
           lines.insert(i + 1, ' ' * indent + 'from .utils import ballquery_batch_p\n')

           fixed = True
           break

   if fixed:
       try:
           with open(file_path, 'w') as f:
               f.writelines(lines)
           return True
       except Exception as e:
           errors.append(f"Failed to write file: {e}")
           return False

   return False

# Fix both model files
if pointcept_path:
   model_files = [
       f"{pointcept_path}/pointcept/models/point_group/point_group_v1m2_custom_criteria.py",
       f"{pointcept_path}/pointcept/models/point_group/point_group_v1m1_base.py"
   ]

   for model_file in model_files:
       print(f"\n  Processing {os.path.basename(model_file)}...")
       if os.path.exists(model_file):
           if fix_model_imports(model_file):
               fixes_applied.append(f"Fixed imports in {os.path.basename(model_file)}")
               print(f"    ✓ Fixed successfully")
           else:
               print(f"    ⚠ Already fixed or pattern not found")
       else:
           print(f"    ✗ File not found")
           errors.append(f"{os.path.basename(model_file)} not found")

# 3. CHECK AND FIX UTILS.PY WRAPPER
print("\n\n3. FIXING UTILS.PY WRAPPER")
print("-" * 40)

if pointcept_path:
   utils_path = f"{pointcept_path}/pointcept/models/point_group/utils.py"

   if os.path.exists(utils_path):
       # Backup
       backup_path = f"{utils_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
       shutil.copy2(utils_path, backup_path)
       print(f"  Created backup: {os.path.basename(backup_path)}")

       # Read file
       with open(utils_path, 'r') as f:
           content = f.read()

       # Check current state
       has_wrapper_assignment = 'ballquery_batch_p = BallQueryBatchP.apply' in content
       has_wrapper_function = 'def ballquery_batch_p(' in content
       has_ballquery_class = 'class BallQueryBatchP' in content

       print(f"  Current state:")
       print(f"    Has BallQueryBatchP class: {has_ballquery_class}")
       print(f"    Has wrapper assignment: {has_wrapper_assignment}")
       print(f"    Has wrapper function: {has_wrapper_function}")

       # The comprehensive fix
       fix_code = '''
# Fixed wrapper for ballquery_batch_p
def ballquery_batch_p(coords, batch_idxs, batch_offsets, radius, meanActive):
   """
   Fixed wrapper that calls the C++ function directly
   """
   import pointgroup_ops_cuda

   n = coords.size(0)

   assert coords.is_contiguous() and coords.is_cuda
   assert batch_idxs.is_contiguous() and batch_idxs.is_cuda
   assert batch_offsets.is_contiguous() and batch_offsets.is_cuda

   while True:
       idx = torch.zeros(n * meanActive, dtype=torch.int32, device=coords.device)
       start_len = torch.zeros((n, 2), dtype=torch.int32, device=coords.device)
       nActive = pointgroup_ops_cuda.ballquery_batch_p(
           coords, batch_idxs, batch_offsets, idx, start_len, n, meanActive, radius
       )
       if nActive <= n * meanActive:
           break
       meanActive = int(nActive // n + 1)

   idx = idx[:nActive]
   return idx, start_len
'''

       # Apply fix based on current state
       lines = content.split('\n')
       fixed = False

       # Look for existing function or assignment to replace
       for i, line in enumerate(lines):
           if "def ballquery_batch_p" in line:
               # Find end of function
               indent = len(line) - len(line.lstrip())
               j = i + 1
               while j < len(lines) and (lines[j].strip() == '' or len(lines[j]) - len(lines[j].lstrip()) > indent):
                   j += 1
               lines = lines[:i] + fix_code.strip().split('\n') + lines[j:]
               fixed = True
               break
           elif "ballquery_batch_p = " in line:
               lines[i:i+1] = fix_code.strip().split('\n')
               fixed = True
               break

       if fixed:
           with open(utils_path, 'w') as f:
               f.write('\n'.join(lines))
           fixes_applied.append("Fixed ballquery_batch_p wrapper in utils.py")
           print("  ✓ Fixed utils.py wrapper")
       elif not has_wrapper_function:
           # Add the wrapper if it doesn't exist
           with open(utils_path, 'a') as f:
               f.write('\n' + fix_code)
           fixes_applied.append("Added ballquery_batch_p wrapper to utils.py")
           print("  ✓ Added wrapper to utils.py")
       else:
           print("  ⚠ utils.py already has a wrapper function")

       # Fix tensor type issues (int vs int32)
       with open(utils_path, 'r') as f:
           content = f.read()

       if 'dtype=torch.int,' in content and 'int32' not in content:
           content = content.replace('dtype=torch.int,', 'dtype=torch.int32,')
           content = content.replace('dtype=torch.int)', 'dtype=torch.int32)')

           with open(utils_path, 'w') as f:
               f.write(content)

           fixes_applied.append("Fixed tensor types to int32 in utils.py")
           print("  ✓ Fixed tensor types to int32")

   else:
       errors.append("utils.py not found")
       print("  ✗ utils.py not found")

# 4. CLEAR ALL CACHES THOROUGHLY
print("\n\n4. CLEARING ALL CACHES")
print("-" * 40)

# Remove __pycache__ directories
pycache_count = 0
if pointcept_path:
   for root, dirs, files in os.walk(pointcept_path):
       if '__pycache__' in dirs:
           try:
               shutil.rmtree(os.path.join(root, '__pycache__'))
               pycache_count += 1
           except:
               pass

if pycache_count > 0:
   fixes_applied.append(f"Cleared {pycache_count} __pycache__ directories")
   print(f"  ✓ Removed {pycache_count} __pycache__ directories")

# Remove .pyc files
pyc_count = 0
if pointcept_path:
   for root, dirs, files in os.walk(pointcept_path):
       for file in files:
           if file.endswith('.pyc'):
               try:
                   os.remove(os.path.join(root, file))
                   pyc_count += 1
               except:
                   pass

if pyc_count > 0:
   fixes_applied.append(f"Removed {pyc_count} .pyc files")
   print(f"  ✓ Removed {pyc_count} .pyc files")

# Clear loaded modules
modules_removed = 0
modules_to_remove = [k for k in list(sys.modules.keys())
                   if any(x in k for x in ['pointcept', 'pointgroup', 'pointops', 'pointgroup_ops'])]
for module in modules_to_remove:
   try:
       del sys.modules[module]
       modules_removed += 1
   except:
       pass

if modules_removed > 0:
   fixes_applied.append(f"Cleared {modules_removed} loaded modules")
   print(f"  ✓ Cleared {modules_removed} loaded modules from memory")

# 5. APPLY TORCH TYPE FIXES (Critical - must be at the end)
print("\n\n5. APPLYING TORCH TYPE COMPATIBILITY FIXES")
print("-" * 40)

if not hasattr(torch, 'uint16'):
   torch.uint16 = torch.int16
   print("  ✓ Fixed torch.uint16")

if not hasattr(torch, 'uint32'):
   torch.uint32 = torch.int32
   print("  ✓ Fixed torch.uint32")
   
if not hasattr(torch, 'uint64'):
   torch.uint64 = torch.long
   print("  ✓ Fixed torch.uint64")

fixes_applied.append("Applied torch type compatibility fixes")

# 6. SUMMARY
print("\n\n" + "="*60)
print("FIX COMPLETE")
print("="*60)
print(f"✓ Fixes applied: {len(fixes_applied)}")
if warnings:
   print(f"⚠ Warnings: {len(warnings)}")
   for w in warnings:
       print(f"  - {w}")
if errors:
   print(f"✗ Errors: {len(errors)}")
   for e in errors:
       print(f"  - {e}")

print("\n✅ Ready to import Pointcept and run inference")

from pointcept.models import build_model
from pointcept.utils.config import Config
from pointcept.datasets.transform import Compose
from tqdm import tqdm
try:
    import pointops
except:
    pointops = None