# PETRv2 Multi-Frame Adaptive Extensions

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0+-red.svg)](https://pytorch.org/)

This repository contains **extensions and improvements** to the original PETRv2 model for multi-view 3D object detection, focusing on **adaptive multi-frame temporal fusion** mechanisms. The work introduces novel adaptive weighting strategies that dynamically adjust the contribution of different temporal frames based on motion patterns and scene dynamics.

## 🚀 Key Contributions

### 1. **Multi-Frame Adaptive Temporal Fusion**
- **3-Frame Adaptive Model**: Extends PETRv2 to process 3 consecutive frames with adaptive weighting
- **4-Frame Adaptive Model**: Further extension to 4 frames for enhanced temporal context
- **Dynamic Weight Assignment**: Implements scene-aware adaptive weights that adjust based on motion patterns

### 2. **Scene-Adaptive Weighting Strategies**
The adaptive mechanism assigns different weights to frames based on scene characteristics:

| Scenario | 3-Frame Weights | 4-Frame Weights | Description |
|----------|----------------|-----------------|-------------|
| **Fast Motion** | [0.55, 0.30, 0.15] | [0.55, 0.25, 0.15, 0.05] | Higher weight on current frame |
| **Moderate Motion** | [0.45, 0.35, 0.20] | [0.40, 0.30, 0.20, 0.10] | Balanced temporal fusion |
| **Static Scene** | [0.40, 0.35, 0.25] | [0.30, 0.28, 0.25, 0.17] | Even distribution across frames |
| **Occlusion** | [0.40, 0.40, 0.20] | [0.25, 0.35, 0.25, 0.15] | Higher weight on previous frames |

### 3. **Comprehensive Visualization Tools**
- **Adaptive Weight Visualization**: Tools to visualize and analyze the adaptive weighting mechanism
- **Multi-Frame Detection Results**: Enhanced visualization for multi-frame 3D object detection
- **Performance Analysis**: Comprehensive metrics comparison and analysis tools

### 4. **Enhanced Training Infrastructure**
- **SLURM Integration**: Optimized training scripts for cluster computing
- **Multi-GPU Support**: Efficient distributed training across multiple GPUs
- **Automated Visualization**: Post-training visualization pipeline

## 📊 Performance Improvements

The adaptive multi-frame extensions demonstrate significant improvements over the baseline PETRv2:

- **Enhanced Temporal Understanding**: Better utilization of temporal information through adaptive weighting
- **Scene-Aware Processing**: Dynamic adaptation to different motion patterns and scene types
- **Improved Detection Accuracy**: Better handling of occlusions and fast-moving objects

## 🛠️ Installation & Setup

### Prerequisites
- Linux (tested on Ubuntu 18.04+)
- Python 3.6.8+
- CUDA 11.2+
- PyTorch 1.9.0+
- mmdetection3d 0.17.1+

### Installation Steps

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/PETRv2-Adaptive-Extensions.git
cd PETRv2-Adaptive-Extensions
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Setup mmdetection3d**:
```bash
cd mmdetection3d
pip install -v -e .
cd ..
```

4. **Prepare dataset**:
Follow the [mmdetection3d data preparation guide](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/data_preparation.md) for nuScenes dataset.

## 🚀 Quick Start

### Training Multi-Frame Adaptive Models

**3-Frame Adaptive Model**:
```bash
bash slurm_train_3frame.sh
```

**4-Frame Adaptive Model**:
```bash
bash slurm_train_4frame.sh
```

### Visualization

**Visualize Adaptive Weights**:
```bash
python tools/misc/visualize_adaptive_weights.py \
    projects/configs/petrv2/petrv2_3frame_adaptive.py \
    work_dirs/petrv2_3frame_adaptive/latest.pth \
    --show-dir adaptive_weights_vis \
    --frames 3 \
    --num-samples 5
```

**Visualize Detection Results**:
```bash
python tools/misc/visualize_3frame_results.py \
    projects/configs/petrv2/petrv2_3frame_adaptive.py \
    --checkpoint work_dirs/petrv2_3frame_adaptive/latest.pth \
    --result work_dirs/petrv2_3frame_adaptive/results.pkl \
    --show-dir vis_results_3frame \
    --frames 3 \
    --score-thr 0.4
```

## 📁 Project Structure

```
PETRv2-Adaptive-Extensions/
├── projects/
│   ├── configs/
│   │   ├── petrv2/
│   │   │   └── petrv2_3frame_adaptive.py    # 3-frame adaptive config
│   │   └── denoise/                         # Additional configurations
│   └── mmdet3d_plugin/                      # Custom model implementations
├── tools/
│   └── misc/
│       ├── visualize_adaptive_weights.py    # Adaptive weight visualization
│       ├── visualize_3frame_results.py     # Multi-frame result visualization
│       └── README_visualization.md         # Visualization documentation
├── slurm_train_3frame.sh                   # 3-frame training script
├── slurm_train_4frame.sh                   # 4-frame training script
├── visualize_petr_4frame.py                 # 4-frame visualization script
├── results.py                              # Results analysis and plotting
└── README.md                               # This file
```

## 🔬 Technical Details

### Adaptive Weighting Mechanism

The adaptive weighting system dynamically adjusts frame contributions based on:

1. **Motion Analysis**: Detects motion patterns in the scene
2. **Temporal Consistency**: Evaluates consistency across frames
3. **Occlusion Handling**: Adjusts weights when objects are occluded
4. **Scene Dynamics**: Adapts to static vs. dynamic scenes

### Model Architecture Extensions

- **Temporal Fusion Module**: Enhanced temporal feature fusion with adaptive weighting
- **Multi-Frame Processing**: Extended input processing for 3-4 consecutive frames
- **Adaptive Attention**: Scene-aware attention mechanisms for temporal features

## 📈 Results & Analysis

The repository includes comprehensive analysis tools:

- **Performance Metrics**: mAP and NDS comparisons across different configurations
- **Adaptive Weight Analysis**: Statistical analysis of weight distributions
- **Visualization Tools**: Multi-frame detection result visualization
- **Training Curves**: Loss and validation metric tracking

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Original PETRv2**: Based on the excellent work by [Liu et al.](https://arxiv.org/abs/2206.01256)
- **mmdetection3d**: Built upon the OpenMMLab detection framework
- **nuScenes Dataset**: Thanks to the nuScenes team for the comprehensive dataset

## 📚 Citation

If you use this work in your research, please cite:

```bibtex   
@article{petrv2_adaptive_extensions,
  title={PETRv2 Multi-Frame Adaptive Extensions: Enhanced Temporal Fusion for 3D Object Detection},
  author={Your Name},
  journal={Your Conference/Journal},
  year={2024}
}
```

## 📞 Contact

For questions and discussions:
- **Email**: ayushgoy@usc.edu
- **Issues**: [GitHub Issues](https://github.com/yourusername/PETRv2-Adaptive-Extensions/issues)

---

**Note**: This repository extends the original PETRv2 implementation with novel adaptive multi-frame temporal fusion mechanisms. The original PETRv2 paper and implementation can be found [here](https://github.com/megvii-research/PETR).