# DIGIT Sensor Simulation and Deep Learning Framework

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive simulation and deep learning framework for DIGIT tactile sensors, featuring bidirectional neural networks for physical-visual mapping and advanced perception capabilities.

## 📊 **Dataset**

The complete dataset for this project is available on Hugging Face:

🔗 **[DIGIT Simulation Dataset](https://huggingface.co/datasets/Ndolphin/DIGIT_simulation/tree/main/datasets)**

### Dataset Contents:
- **NodalDataOutput.zip**: CSV files containing nodal displacement data [x, y, dz] from FEM simulations
- **DeltaImages_100x75.zip**: RGB delta images (100×75 resolution) showing tactile sensor changes
- **ContactMasks_100x75.zip**: Binary contact masks indicating interaction regions

### Dataset Statistics:
- **Sensors**: 3 different DIGIT sensors (D21119, D21242, D21273)
- **Objects**: 7+ different objects (hammar, mug, Rabit_L, Rabit_R, scissor, soupcan, stamp)
- **Time Steps**: ~100 interaction sequences per object
- **Total Samples**: ~150+ paired CSV-image samples for training
- **Test Data**: 8 unseen object configurations (id1-id8) with 100 steps each

### Usage:
```python
# Download dataset using Hugging Face datasets library
from huggingface_hub import snapshot_download

# Download entire dataset
snapshot_download(repo_id="Ndolphin/DIGIT_simulation", repo_type="dataset", local_dir="./DIGIT_data")

# Or download specific files
from huggingface_hub import hf_hub_download

# Download nodal data
nodal_data = hf_hub_download(repo_id="Ndolphin/DIGIT_simulation", filename="datasets/NodalDataOutput.zip", repo_type="dataset")
```

## Overview

This repository contains a complete pipeline for DIGIT tactile sensor simulation and analysis, including:

- **FEM Simulation**: SOFA-based finite element modeling of tactile interactions
- **Perception Network**: Visual-to-physical displacement mapping using U-Net
- **Rendering Network**: Physical-to-visual delta image synthesis using UNetSmall
- **Sensor Experiments**: Real DIGIT sensor data collection and processing

## Architecture

### 1. **Perception Network** (Visual → Physical)
- **Input**: RGB delta images (320×240×3)
- **Output**: Physical displacement fields [x, y, dz] (320×240×3)
- **Architecture**: Clean U-Net with skip connections
- **Purpose**: Extract physical sensor data from visual representations

### 2. **Rendering Network** (Physical → Visual)
- **Input**: Multi-channel sensor data (19×75×100)
  - Displacement field (1 channel)
  - Coordinate grids (2 channels)
  - Fourier positional encoding (16 channels)
- **Output**: RGB delta images (3×75×100)
- **Architecture**: UNetSmall with masked loss
- **Purpose**: Generate realistic tactile visualizations from sensor data

### 3. **FEM Simulation**
- SOFA-based finite element modeling
- Contact simulation with various indenters
- Nodal displacement data generation

## Repository Structure

```
DIGIT_simulation/
├── FEM_simulation_SOFAscene/          # SOFA-based FEM simulation
│   ├── ContactTest_in_loop.py         # Batch contact simulation
│   ├── ContactTest_single.py          # Single contact test
│   ├── meshes/                        # 3D mesh files for simulation
│   └── NodalDataOutput/               # Generated CSV nodal data
├── Perception_Network/                # Visual-to-physical mapping
│   ├── Train.py                       # Training script for perception
│   ├── data_preprocessing.py          # Data preparation utilities
│   ├── Testset_inference.py          # Inference on test data
│   └── perception_unet_model.pth      # Trained model weights
├── Rendering_Network/                 # Physical-to-visual synthesis
│   ├── Train.py                       # Training script for rendering
│   ├── data_loader.py                 # Data loading utilities
│   ├── inference.py                   # Inference script
│   └── best_model.pt                  # Trained model weights
├── DIGIT_sensor_experiment/           # Real sensor experiments
│   ├── DIGIT_CameraView.py           # Camera interface
│   ├── ImageExtraction.py            # Image processing
│   └── Recorded_Videos/              # Experimental data
└── requirements.txt                   # Python dependencies
```

## Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/ndolphin-github/DIGIT_simulation.git
cd DIGIT_simulation
```

2. **Create a virtual environment:**
```bash
python -m venv digit_env
source digit_env/bin/activate  # On Windows: digit_env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Networks

#### Perception Network (Visual → Physical)
```bash
cd Perception_Network
python Train.py --epochs 50 --batch_size 4 --learning_rate 1e-3
python Testset_inference.py --model_path perception_unet_model.pth
```

#### Rendering Network (Physical → Visual)
```bash
cd Rendering_Network
python Train.py --epochs 50 --batch_size 8 --lr 2e-4
python inference.py -m best_model.pt -u NodalDataOutput -o generated_images
```

#### FEM Simulation
```bash
cd FEM_simulation_SOFAscene
python ContactTest_single.py  # Single contact simulation
python ContactTest_in_loop.py # Batch simulation
```

## Data Format

### Dataset Structure
The complete dataset is available at: **[Hugging Face DIGIT Simulation Dataset](https://huggingface.co/datasets/Ndolphin/DIGIT_simulation/tree/main/datasets)**

```
DIGIT_simulation_dataset/
├── datasets/
│   ├── NodalDataOutput.zip          # Nodal displacement data
│   │   ├── D21119/                  # Sensor ID
│   │   │   ├── hammar/              # Object name
│   │   │   │   ├── topROI_step_000.csv
│   │   │   │   ├── topROI_step_001.csv
│   │   │   │   └── ...
│   │   │   ├── mug/
│   │   │   └── ...
│   │   ├── D21242/
│   │   └── D21273/
│   ├── DeltaImages_100x75.zip       # Tactile visualization images
│   │   ├── D21119/
│   │   │   ├── hammar/
│   │   │   │   ├── hammar_000.jpg
│   │   │   │   ├── hammar_001.jpg
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   └── ContactMasks_100x75.zip      # Binary contact masks
│       ├── D21119/
│       │   ├── hammar/
│       │   │   ├── hammar_000_mask.jpg
│       │   │   ├── hammar_001_mask.jpg
│       │   │   └── ...
│       │   └── ...
│       └── ...
```

### Input Data
- **CSV Files**: Nodal displacement data with columns [x, y, dz]
- **RGB Images**: Delta images showing tactile sensor changes
- **Contact Masks**: Binary masks indicating contact regions

### Coordinate System
- **X Range**: [-7.5mm, +7.5mm] (sensor width)
- **Y Range**: [0mm, 20mm] (sensor height)
- **Resolution**: 100×75 pixels (rendering), 320×240 pixels (perception)

### Data Loading Example
```python
import pandas as pd
from PIL import Image
import numpy as np

# Load nodal data
df = pd.read_csv("topROI_step_000.csv")
x_coords = df["x"].values  # Node X coordinates
y_coords = df["y"].values  # Node Y coordinates  
dz_values = df["dz"].values  # Displacement values

# Load corresponding delta image
delta_img = Image.open("hammar_000.jpg")
delta_array = np.array(delta_img)  # Shape: (75, 100, 3)

# Load contact mask
mask_img = Image.open("hammar_000_mask.jpg")
mask_array = np.array(mask_img) > 0  # Binary mask
```

## Model Details

### Hyperparameters

| Parameter | Perception Network | Rendering Network |
|-----------|-------------------|-------------------|
| Input Size | 320×240×3 | 19×75×100 |
| Output Size | 320×240×3 | 3×75×100 |
| Batch Size | 4 | 8 |
| Learning Rate | 1e-3 | 2e-4 |
| Optimizer | Adam | AdamW |
| Loss Function | MSE (dz only) | Masked L1 |
| Epochs | 50 | 50 |

### Loss Functions

**Perception Network:**
```python
loss = MSE(predicted_dz, target_dz)  # Focus on displacement only
```

**Rendering Network:**
```python
loss = contact_loss + 0.1 * background_loss + 0.02 * neutrality_loss
```

## Performance

### Inference Speed (RTX 5090)
- **Perception Network**: ~0.35ms per image (FP32), ~0.22ms (FP16)
- **Rendering Network**: ~0.20ms per image (FP32), ~0.12ms (FP16)
- **Throughput**: 2k-10k+ images/second depending on batch size

### Model Complexity
- **Perception Network**: ~500K-1M parameters
- **Rendering Network**: ~200K parameters (lightweight design)

## 📈 Results

The framework enables:
- **High-fidelity** tactile simulation and reconstruction
- **Real-time** inference capabilities
- **Bidirectional** mapping between physical and visual domains
- **Robust** generalization to unseen objects and contact scenarios

## 🔬 Research Applications

- **Tactile Sensing**: Enhanced understanding of touch-based perception
- **Robotics**: Improved manipulation through tactile feedback
- **Material Science**: Analysis of contact mechanics and deformation
- **Human-Computer Interaction**: Natural touch interfaces

##  Citation

<!-- If you use this work in your research, please cite:

```bibtex
@article{digit_simulation_2025,
  title={DIGIT Sensor Simulation and Deep Learning Framework},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2025},
  note={Available at: https://github.com/ndolphin-github/DIGIT_simulation}
}
``` -->



## Acknowledgments

- SOFA Framework for FEM simulation capabilities


## Contact

For questions and support, please open an issue on GitHub or contact [ndolphin93@gmail.com].

