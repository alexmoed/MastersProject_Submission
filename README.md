#SegSplat A 3D Gaussian Splat Segmentation Pipeline

A multi-stage segmentation pipeline explicitly designed for 3D Gaussian splats, combining Meta's SONATA pre-trained model with PointGroup instance segmentation decoder. This project bridges the challenging domain gap between RGB-D trained models and real-world Gaussian splat data from photogrammetric reconstruction.

## Overview

This pipeline addresses the unique challenges of segmenting Gaussian splats that differ from the RGB-D data used for training. Unlike RGB-D captures with uniform density and minimal noise, Gaussian splats contain:
- Varying point densities across the scene
- Reconstruction artifacts from photogrammetry
- Inconsistent point distributions
- No standardized evaluation datasets

The multi-stage approach processes splats through semantic and instance segmentation models trained on ScanNet, carefully tuned to handle the noisy, irregular nature of photogrammetric reconstructions while preserving the original splat attributes including spherical harmonics.

## Key Features

- **Designed for Gaussian Splats**: Handles varying densities and reconstruction artifacts inherent in photogrammetric data
- **Strategic Multi-Model Approach**:
  - ScanNet 20 (0.021m grid) excels at large furniture with higher confidence
  - ScanNet 200 (0.010m grid) captures small objects
  - Dedicated semantic wall/floor (0.020m grid) model designed to avoid ceiling misclassification issues
  - Pipeline preserves the best predictions from each specialized model
- **Non-destructive Processing**: Original PLY remains untouched with all spherical harmonics preserved, classifications added as new attributes
- **Domain Transfer**: This method can generalize from RGB-D training to noisy splats through augmentation and parameter tuning

## Performance

**ScanNet v2 Benchmark Results**

Individual model performance (ScanNet 20 checkpoint only):
- SONATA + PointGroup: 64.4% mAP@0.5
- Original PointGroup: 63.6% mAP@0.5

*Note: These metrics come from RGB-D evaluation on ScanNet. We do not have any annotated Gaussian splat datasets available for quantitative evaluation of the full pipeline. The demo videos and interactive visualizations showcase the visual results on splats.*

## Installation

### Requirements

#### Hardware
- CUDA-capable GPU (NVIDIA A100 40GB recommended)
- Minimum 40GB VRAM for training
- 16GB+ VRAM for inference

#### Software
- Python 3.11
- CUDA 11.8
- PyTorch 2.1.0+cu118
- NumPy 1.24.0 (specific version required)

### Setup Steps

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

### Google Colab
The project includes installation scripts designed for Google Colab:
- `setup_cuda_torch.py`: Handles CUDA and PyTorch installation with caching
- `setup_build_env.py`: Sets up the build environment and dependencies
- Note: NumPy 1.24.0 compatibility fixes are automatically applied

## Usage

### Notebooks

The repository includes three Google Colab notebooks:

- **Training.ipynb** - Multiple training configurations:
  - Instance segmentation: PointGroup decoder with ScanNet 20/200
  - Semantic segmentation: SONATA decoder for full scene or wall/floor only
  - Custom 2-class dataset remapping script (wall=0, floor=1, other=-1)
  - Pre-configured training commands with optimal parameters

- **Evaluation.ipynb**: Evaluate model performance on ScanNet validation set (RGB-D data). Note: No annotated Gaussian splat dataset exists for evaluation

- **Inference_pipeline.ipynb**: Run the complete segmentation pipeline on Gaussian splats

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexmoed/MastersProject_Submission/blob/main/scripts/Inference_pipeline.ipynb)

### Pipeline Stages

1. **Preprocessing**: Temporarily converts spherical harmonics to RGB values and scales coordinates to match training data for preview and predictions. Original PLY files are not modified.

2. **Instance Segmentation (ScanNet 20)**: Identifies major furniture and large objects using voxelization with 0.021m grid size.

3. **Semantic Segmentation**: Detects walls and floors while preserving high-confidence predictions from previous stages.

4. **Fine-grained Segmentation (ScanNet 200)**: Adds detailed object classification for smaller items without overwriting large object predictions.

5. **Export**: Compiles results and exports PLY files that retain the original spherical harmonics, scale, and point count with segmentation attributes for use in 3D software such as Houdini.

### Output Format

Final PLY files include four new attributes per point, but each point receives classification from only ONE model:

**If classified by ScanNet 20** (large objects/walls/floors):
- `ScanNet20_class`: Object class ID
- `ScanNet20_instance`: Instance ID  
- `ScanNet200_class`: -1
- `ScanNet200_instance`: -1

**If classified by ScanNet 200** (small objects):
- `ScanNet20_class`: -1
- `ScanNet20_instance`: -1
- `ScanNet200_class`: Object class ID
- `ScanNet200_instance`: Instance ID

**If unclassified**:
- All four attributes set to -1

The pipeline ensures no overlap - each point is classified by either ScanNet 20, ScanNet 200, or remains unclassified.

## Models & Data

### Pre-trained Models
- **SONATA Base Encoder**: [Download from Hugging Face](https://huggingface.co/facebook/sonata/resolve/main/pretrain-sonata-v1m1-0-base.pth)
  - Meta's pre-trained encoder checkpoint (PTv3 architecture)
  - Self-supervised training on 140K scenes
  - Foundation model for all downstream tasks

### Fine-tuned Models
- **ScanNet 20 Instance Segmentation**: [Download](https://storage.googleapis.com/anmstorage/MasterProject/insseg-pointgroup-sonata_3/epoch_1200.pth)
  - PointGroup instance segmentation head
  - Trained for large furniture and major objects
  - 20 object categories

- **ScanNet 200 Instance Segmentation**: [Download](https://storage.googleapis.com/anmstorage/MasterProject/insseg-pointgroup-sonata_config_200_1/epoch_800.pth)
  - PointGroup instance segmentation head
  - Fine-grained object detection
  - 200 object categories including smaller items

- **Full Scene Semantic Segmentation**: [Download](https://storage.googleapis.com/anmstorage/MasterProject/semseg-sonata-v1m1-0c-scannet-ft-_full_scene_v003/epoch_800.pth)
  - SONATA semantic segmentation head
  - Wall and floor detection
  - Full scene understanding

### Dataset Access
To access the ScanNet dataset for training:
1. Request access at [Hugging Face ScanNet Repository](https://huggingface.co/datasets/Pointcept/scannet-compressed) (preprocessed Pointcept format)
2. Agree to the terms of use
3. Download and extract to your data directory

Note: ScanNet access requires agreeing to non-commercial research terms.

### Custom Training Options
- **2-Class Training**: For wall/floor-only models, use the included dataset remapping script that converts ScanNet to 2 classes (wall=0, floor=1, other=-1)
- **Custom Class Selection**: Modify the remapping script in `Training.ipynb` to train on your specific classes of interest

## Demos

### Interactive 3D Visualizations

**Note**: USD files are interactive 3D displays. Please allow a few minutes for loading.

#### RGB-D Examples
- **RGB-D Example 1**: [View Interactive Demo](https://autodesk-forks.github.io/USD/usd_for_web_demos/test.html?file=https://storage.googleapis.com/anmstorage/MasterProject/USD/RGB_D_example_v001.usd)
- **RGB-D Example 2**: [View Interactive Demo](https://autodesk-forks.github.io/USD/usd_for_web_demos/test.html?file=https://storage.googleapis.com/anmstorage/MasterProject/USD/RGB_D_example_v002.usd)

#### Gaussian Splat Scenes
- **Living Room Example**: [View Interactive Demo](https://autodesk-forks.github.io/USD/usd_for_web_demos/test.html?file=https://storage.googleapis.com/anmstorage/MasterProject/USD/living_room_only_splat_v001.usd)

### Videos
- **Kitchen Classification**: [View Video](https://storage.googleapis.com/anmstorage/MasterProject/demo/kitchen_classification.mp4)
- **Living Room Classification**: [View Video](https://storage.googleapis.com/anmstorage/MasterProject/demo/living_room_classification.mp4)

## Input Files

### Sample PLY Files (Unclassified)
Raw Gaussian splat files to test with the inference pipeline:
- [Full Kitchen/Diner Scene](https://storage.googleapis.com/anmstorage/MasterProject/PLY_Files/KitchenDiner_cleaned_v014_rotation.ply)
- [Kitchen Only](https://storage.googleapis.com/anmstorage/MasterProject/PLY_Files/KitchenDiner_cleaned_v010_rotation_just_kitchen.ply)

Use these files with `Inference_pipeline.ipynb` to test the segmentation pipeline.

## Applications

- VFX asset extraction and manipulation
- 3D scene understanding
- Object isolation for relighting
- Building reusable 3D asset libraries
- Selective mesh conversion
- Per-object attribute modification

## Citation

This work builds upon the following foundational research:

**SONATA** - Meta's pre-trained model checkpoint we use as the foundation for feature extraction:
```bibtex
@inproceedings{wu2025sonata,
    title={Sonata: Self-Supervised Learning of Reliable Point Representations},
    author={Wu, Xiaoyang and DeTone, Daniel and Frost, Duncan and Shen, Tianwei and Xie, Chris and Yang, Nan and Engel, Jakob and Newcombe, Richard and Zhao, Hengshuang and Straub, Julian},
    booktitle={CVPR},
    year={2025}
}
```

**Point Transformer V3** - The encoder architecture used by SONATA:
```bibtex
@inproceedings{wu2024ptv3,
    title={Point Transformer V3: Simpler, Faster, Stronger},
    author={Wu, Xiaoyang and Jiang, Li and Wang, Peng-Shuai and Liu, Zhijian and Liu, Xihui and Qiao, Yu and Ouyang, Wanli and He, Tong and Zhao, Hengshuang},
    booktitle={CVPR},
    year={2024}
}
```

**PointGroup** - The instance segmentation decoder head we combined with SONATA encoder:
```bibtex
@inproceedings{jiang2020pointgroup,
    title={PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation},
    author={Jiang, Li and Zhao, Hengshuang and Shi, Shaoshuai and Liu, Shu and Fu, Chi-Wing and Jia, Jiaya},
    booktitle={CVPR},
    year={2020}
}
```

**Pointcept** - The comprehensive codebase that houses all the methods used in this work:
```bibtex
@misc{pointcept2023,
    title={Pointcept: A Codebase for Point Cloud Perception Research},
    author={Pointcept Contributors},
    howpublished = {\url{https://github.com/Pointcept/Pointcept}},
    year={2023}
}
```
