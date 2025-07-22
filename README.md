# DIGIT Sensor Simulation Project

This project implements a comprehensive simulation pipeline for DIGIT tactile sensors, including sensor data processing, finite element method (FEM) simulation, perception networks, and reconstruction algorithms.

## Project Structure

```
DIGIT_simulation/
├── DIGIT_sensor_experiment/    # Sensor data processing and video analysis
├── FEM_simulation/            # Finite Element Method simulations
├── Perception_Network/        # Neural networks for sensor perception
└── Reconstruction_Network/    # Image reconstruction algorithms
```

## Components

### 1. DIGIT Sensor Experiment
- **DIGIT_CameraView.py**: Camera interface and view management
- **ImageExtraction.py**: Extract frames from video data
- **Video_CenterPoint.py**: Analyze center points in videos
- **EditedVideos/**: Collection of processed video files
- **output/**: Extracted image frames

### 2. FEM Simulation
- **ContactTest_Live.py**: Real-time contact testing
- **ContactTest_loop.py**: Batch contact testing
- **indenter_settings.json**: Configuration for standard indenters
- **Unseen_indenter_settings.json**: Configuration for novel indenters
- **meshes/**: 3D mesh files for simulation

### 3. Perception Network
- **train_rgb2dz.py**: Train RGB to depth displacement mapping
- **rgb2dz_dataset.py**: Dataset loading utilities

### 4. Reconstruction Network
- *(Coming soon)*

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DIGIT_simulation.git
cd DIGIT_simulation

# Install dependencies
pip install torch opencv-python numpy
```

## Usage

### Training Perception Network
```bash
cd Perception_Network
python train_rgb2dz.py
```

### Running FEM Simulation
```bash
cd FEM_simulation
python ContactTest_Live.py
```

## Dataset

Due to the large size of the dataset (15GB+), the training data is hosted separately:
- **Large dataset**: [Link to Hugging Face or other hosting service]
- **Sample data**: Included in repository for testing

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- (Add other dependencies as needed)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license here]

## Citation

If you use this code in your research, please cite:
```
[Add citation information]
```
