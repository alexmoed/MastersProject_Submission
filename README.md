# 3D Gaussian Splat Segmentation Pipeline

A multi-stage segmentation pipeline for 3D Gaussian splats using Meta's SONATA encoder combined with PointGroup instance segmentation. This project bridges the gap between RGB-D trained models and real-world Gaussian splat data from photogrammetric reconstruction.

## Overview

This pipeline processes 3D Gaussian splats through multiple segmentation stages to identify and classify objects in 3D scenes. It combines semantic and instance segmentation models trained on ScanNet datasets to provide comprehensive scene understanding.

## Model Checkpoints

### Pre-trained Models
- **SONATA Base Checkpoint**: [Download](https://huggingface.co/facebook/sonata/resolve/main/pretrain-sonata-v1m1-0-base.pth)
  - Meta's pre-trained encoder (PTv3 architecture)
  - Self-supervised training on 140K scenes
  
### Fine-tuned Models
- **ScanNet 20 Instance Segmentation**: [Download](https://storage.googleapis.com/anmstorage/MasterProject/insseg-pointgroup-sonata_3/epoch_1200.pth)
  - Trained for large furniture and major objects
  - 20 object categories
  
- **ScanNet 200 Instance Segmentation**: [Download](https://storage.googleapis.com/anmstorage/MasterProject/insseg-pointgroup-sonata_config_200_1/epoch_800.pth)
  - Fine-grained object detection
  - 200 object categories including smaller items
  
- **Full Scene Semantic Segmentation**: [Download](https://storage.googleapis.com/anmstorage/MasterProject/semseg-sonata-v1m1-0c-scannet-ft-_full_scene_v003/epoch_800.pth)
  - Wall and floor detection
  - Full scene understanding

## Interactive 3D Demos

**Note**: USD files are interactive 3D displays. Please allow a few minutes for loading.

### RGB-D Examples
- **RGB-D Example 1**: [View Interactive Demo](https://autodesk-forks.github.io/USD/usd_for_web_demos/test.html?file=https://storage.googleapis.com/anmstorage/MasterProject/USD/RGB_D_example_v001.usd)
- **RGB-D Example 2**: [View Interactive Demo](https://autodesk-forks.github.io/USD/usd_for_web_demos/test.html?file=https://storage.googleapis.com/anmstorage/MasterProject/USD/RGB_D_example_v002.usd)

### Gaussian Splat Scenes
- **Living Room Example**: [View Interactive Demo](https://autodesk-forks.github.io/USD/usd_for_web_demos/test.html?file=https://storage.googleapis.com/anmstorage/MasterProject/USD/living_room_only_splat_v001.usd)

## Turntable Videos

### Classification Demonstrations
- **Kitchen Classification**: [View Video](https://storage.googleapis.com/anmstorage/MasterProject/demo/kitchen_classification.mp4)
- **Living Room Classification**: [View Video](https://storage.googleapis.com/anmstorage/MasterProject/demo/living_room_classification.mp4)

## Sample PLY Files

### Kitchen/Diner Scenes
- [Full Kitchen/Diner Scene](https://storage.googleapis.com/anmstorage/MasterProject/PLY_Files/KitchenDiner_cleaned_v014_rotation.ply)
- [Kitchen Only](https://storage.googleapis.com/anmstorage/MasterProject/PLY_Files/KitchenDiner_cleaned_v010_rotation_just_kitchen.ply)

## Pipeline Stages

### Stage 1: Preprocessing
Converts Gaussian splat data to compatible format. Handles spherical harmonics to RGB conversion and coordinate normalization.

### Stage 2: Instance Segmentation (ScanNet 20)
Identifies major furniture and large objects using voxelization with 0.021m grid size.

### Stage 3: Semantic Segmentation
Detects walls and floors while preserving high-confidence predictions from previous stages.

### Stage 4: Fine-grained Segmentation (ScanNet 200)
Adds detailed object classification for smaller items without overwriting large object predictions.

### Stage 5: Export
Compiles results and exports PLY files with segmentation attributes for use in 3D software like Houdini.

## Key Features

- **Multi-stage Processing**: Combines different models for comprehensive segmentation
- **Domain Adaptation**: Bridges RGB-D training data to real Gaussian splats
- **Modular Architecture**: SONATA encoder with task-specific decoder heads
- **Preserves Original Data**: Maintains spherical harmonics and point attributes
- **Flexible Export**: Outputs PLY files with classification attributes

## Requirements

### Hardware Requirements
- CUDA-capable GPU (NVIDIA A100 40GB recommended)
- Minimum 40GB VRAM for training
- 16GB+ VRAM for inference

### Software Requirements
- Python 3.11
- CUDA 11.8
- PyTorch 2.1.0+cu118
- NumPy 1.24.0 (specific version required)

### Core Dependencies
```bash
# Base requirements
torch==2.1.0+cu118
numpy==1.24.0
plotly
scipy
trimesh
plyfile
```

### Installation Steps

1. **Install CUDA 11.8**
   - Follow the setup_cuda_torch.py script for automated installation
   - Includes caching for faster reinstalls on Google Colab

2. **Install PyTorch with CUDA support**
   ```bash
   pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install NumPy (specific version required)**
   ```bash
   pip uninstall -y numpy
   pip install numpy==1.24.0 --no-cache-dir
   ```

4. **Install PointOps (run twice for proper setup)**
   ```bash
   cd Pointcept/libs/pointops
   pip install -v -e .
   # Run the command again
   pip install -v -e .
   ```

5. **Install additional dependencies**
   ```bash
   cd /path/to/project/Installs
   python setup_build_env.py
   ```

### Google Colab Setup
The project includes optimized installation scripts for Google Colab:
- `setup_cuda_torch.py`: Handles CUDA and PyTorch installation with caching
- `setup_build_env.py`: Sets up the build environment and dependencies
- Note: NumPy 1.24.0 compatibility fixes are automatically applied

## Technical Details

- **Architecture**: SONATA (PTv3) encoder + PointGroup decoder
- **Training**: A100 GPU, mixed precision training
- **Point Processing**: Handles varying densities and reconstruction artifacts
- **Coordinate System**: Automatic normalization and scaling for model compatibility

## Output Format

Final PLY files include four attributes per point:
- `ScanNet20_class`: Large object classification
- `ScanNet20_instance`: Instance ID for ScanNet 20 objects
- `ScanNet200_class`: Fine-grained object classification  
- `ScanNet200_instance`: Instance ID for ScanNet 200 objects

Value of -1 indicates no classification for that category.

## Applications

- VFX asset extraction and manipulation
- 3D scene understanding
- Object isolation for relighting
- Building reusable 3D asset libraries
- Selective mesh conversion
- Per-object attribute modification

## Citation

This work builds on Meta's SONATA and leverages the PointGroup instance segmentation framework for processing 3D Gaussian splats.

If you use this work in your research, please cite the following papers:

```bibtex
@misc{pointcept2023,
    title={Pointcept: A Codebase for Point Cloud Perception Research},
    author={Pointcept Contributors},
    howpublished = {\url{https://github.com/Pointcept/Pointcept}},
    year={2023}
}

@inproceedings{wu2024ptv3,
    title={Point Transformer V3: Simpler, Faster, Stronger},
    author={Wu, Xiaoyang and Jiang, Li and Wang, Peng-Shuai and Liu, Zhijian and Liu, Xihui and Qiao, Yu and Ouyang, Wanli and He, Tong and Zhao, Hengshuang},
    booktitle={CVPR},
    year={2024}
}

@inproceedings{wu2025sonata,
    title={Sonata: Self-Supervised Learning of Reliable Point Representations},
    author={Wu, Xiaoyang and DeTone, Daniel and Frost, Duncan and Shen, Tianwei and Xie, Chris and Yang, Nan and Engel, Jakob and Newcombe, Richard and Zhao, Hengshuang and Straub, Julian},
    booktitle={CVPR},
    year={2025}
}
```
