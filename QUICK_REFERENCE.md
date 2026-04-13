# AsyncVLA: Quick Reference Guide

## Paper Summary

**Title:** AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge

**Authors:** Noriaki Hirose¹,², Catherine Glossop¹, Dhruv Shah³, Sergey Levine¹

**Affiliations:**
- ¹ UC Berkeley (BAIR - Berkeley AI Research)
- ² Toyota Motor North America
- ³ Princeton University

**Citation:**
```bibtex
@misc{hirose2026asyncvla,
      title={AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge},
      author={Noriaki Hirose and Catherine Glossop and Dhruv Shah and Sergey Levine},
      year={2026},
      eprint={2602.13476},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.13476}
}
```

---

## Problem & Motivation

### Challenge
Traditional Vision-Language-Action (VLA) models suffer from:
- **High latency** (200-500ms) due to end-to-end inference
- **Computational bottleneck** on edge devices
- **Poor deployment** on robot edge controllers

### Solution
**AsyncVLA** decouples computation:
- **Remote Base VLA** → Expensive computation (GPU server)
- **Edge Adapter** → Real-time adaptation (robot computer)
- **Asynchronous Protocol** → Handles network delays gracefully

**Result:** 3-5× latency reduction (60-130ms end-to-end)

---

## Architecture at a Glance

```
Remote Workstation          Network Link           Robot Edge Controller
─────────────────          ────────────           ──────────────────────

┌──────────────────┐                            ┌──────────────────┐
│ Vision Encoder   │                            │ Edge Adapter     │
│ (DINOv2)         │                            │ (Multi-head Att) │
└────────┬─────────┘                            └────────┬─────────┘
         │                                               │
         ↓                                               ↑
┌──────────────────┐                            ┌──────────────────┐
│ Language Model   │                            │ Action Decoder   │
│ (LLaMA-7B)       ├─────────ROS Bridge─────→  │ (L1 Regression)  │
└──────────────────┘   (tokens: 8 actions)     └──────────────────┘
         │
    LoRA Adapters
    (Trainable)
```

---

## Key Components

| Component | Details | Purpose |
|-----------|---------|---------|
| **Vision Backbone** | DINOv2 ViT or CLIP | Encodes images to embeddings |
| **Language Backbone** | LLaMA-2 7B | Generates action sequences |
| **Proprio Projector** | Linear layers | Maps goal pose to token space |
| **Edge Adapter** | 2-layer Multi-Head Attention | Real-time refinement on robot |
| **Action Head** | L1 Regression or Token Projection | Decodes final actions |

---

## Model Specifications

### Dimensions

```python
# Input
input_ids:           [batch, seq_len]
pixel_values:        [batch, 2, 3, 224, 224]    # 2 images
goal_pose:           [batch, 4]                 # (cos θ, sin θ, Δx, Δy)

# Processing
num_patches:         392 (196 per image)
patch_dim:           768 → 4096 (projected)
seq_len_total:       393 + seq_len (patches + goal + text)

# Output
actions:             [batch, 8, 4]              # 8 steps, 4 dims
                     Δx, Δy, Δθ, velocity
```

### Parameters

```python
Vision Backbone:     ~86M (frozen)
Language Model:      ~7B (LoRA trainable: 5-10M)
Edge Adapter:        ~5-10M (fully trainable)
Total Trainable:     ~15-20M
```

---

## Training Overview

### Multi-Dataset Setup
- **GNM** (Goal Navigation): Image-to-image navigation
- **LeLaN** (Learning to Drive): Legged locomotion
- **SACSoN** (Social Navigation): Crowd-aware planning

### Configuration
```yaml
Training Mode:       Edge adapter + Action head only
Batch Size:          6 + 6 + 6 = 18 (3 datasets)
GPU Count:           1-2× A100 (not H100/H200)
Training Time:       ~24-48 hours
Learning Rate:       1e-4 (with decay at 750k steps)
Optimizer:           AdamW + LoRA
Loss:                Multi-task (5 supervision signals)
```

### Loss Components
1. **L2_actions** - Regression on action outputs
2. **L2_pose_img** - Supervision from pose + image
3. **L2_img** - Image-only supervision
4. **L2_lan** - Language-only supervision
5. **L2_lan_pose** - Language + pose supervision

---

## Inference Pipeline

### Step-by-Step (100ms cycle)

```
Time    Operation                      Latency
────────────────────────────────────────────────
0ms     • Capture camera image         5-10ms
        • Read IMU/GPS

5ms     • Image preprocessing          2-5ms
        • Stack past frame

7ms     • Serialize + send to base     5-20ms
        └──→ Transmit over network

17ms    • Base VLA processes           30-50ms
        • Vision encoding
        • LLM forward pass
        • Token generation

47ms    • Network transmit results     5-20ms
        └──→ Receive at edge

57ms    • Edge adapter processes       10-15ms
        • Multi-head attention
        • Action decoding
        • Denormalization

72ms    • PD controller filtering      5ms
        • Generate motor commands

77ms    • Send PWM signals to motors   5-10ms

85ms    ✓ READY FOR NEXT FRAME
        (15ms buffer before 100ms mark)

TOTAL: 85ms (cf. 200-500ms for end-to-end)
```

### Asynchronous Execution

**Key Innovation:** Edge processes while base VLA computes
- Frame N sent to base at t=5ms
- Edge processes frame N-1 predictions at t=7-15ms
- Base VLA returns at t=37ms
- Edge receives at t=47ms, finishes at t=57ms
- **Result:** Smooth pipeline with minimal idle time

---

## Datasets & Configuration

### Dataset Paths
```yaml
GNM:
  image:   ./data/gnm/images
  pickle:  ./data/gnm/pickles

LeLaN:
  image:   ./data/lelan/images
  pickle:  ./data/lelan/pickles

SACSoN:
  image:   ./data/sacson/images
  pickle:  ./data/sacson/pickles
```

### Data Augmentation
- Random crop (GNM)
- Horizontal flip (GNM)
- Normalize (all)
- Image size: 224×224

---

## Code Organization

```
AsyncVLA/
├── prismatic/              # VLA framework
│   ├── models/
│   │   ├── vlas/openvla.py          # OpenVLA wrapper
│   │   ├── action_heads.py          # Decoding heads
│   │   ├── projectors.py            # Projectors
│   │   └── backbones/               # Vision/Language
│   ├── vla/
│   │   ├── action_tokenizer.py      # Discretization
│   │   └── datasets/                # Data loaders
│   └── extern/hf/                   # HuggingFace integration
├── vla-scripts/
│   └── train_asyncvla.py            # Main training script
├── inference/
│   ├── run_asyncvla.py              # Production inference
│   └── run_asyncvla_v0.py           # Legacy version
├── experiments/robot/
│   └── openvla_utils.py             # Robot utilities
└── config_nav/
    ├── dataset_config.yaml
    └── mbra_config.yaml
```

---

## Performance Metrics

### Latency Breakdown
```
Component          Latency      Parallelizable?
─────────────────────────────────────────────
Sensor capture     5-10ms       ✓ (parallel)
Preprocessing      2-5ms        ✗ (sequential)
Network send       5-20ms       ✓ (async)
Base VLA compute   30-50ms      ✓ (parallel)
Network recv       5-20ms       ✓ (async)
Edge processing    10-15ms      ✓ (parallel)
Motor command      5-10ms       ✗ (sequential)
─────────────────────────────────────────────
TOTAL              60-130ms     (vs 200-500ms)
```

### Memory Usage
```
Component                    Memory
──────────────────────────────────
Vision Backbone (DINOv2)     ~1GB
Language Model (LLaMA-7B)    ~13GB
Edge Adapter                 ~50MB
Activations + Cache          ~1-2GB
PyTorch Overhead             ~1-2GB
──────────────────────────────────
Total (inference)            ~16-19GB
```

---

## Key Hyperparameters

```python
# Model
num_images_in_input = 2              # Current + goal
llm_dim = 4096                       # LLaMA embedding
action_dim = 4                       # [Δx, Δy, Δθ, v]
num_actions_chunk = 8                # Predict 8 steps

# Training
learning_rate = 1e-4
weight_decay = 0.01
grad_accumulation_steps = 2
lora_rank = 64
lora_dropout = 0.05

# Edge Adapter
obs_encoding_size = 1024
mha_num_attention_heads = 8
mha_num_attention_layers = 2
mha_ff_dim_factor = 4

# Data
batch_size_gnm = 6
batch_size_lelan = 6
batch_size_sacson = 6
image_size = 224
waypoint_spacing = 1
context_size = 5
```

---

## Hardware Requirements

### For Training Base VLA
- **GPU:** 5× Nvidia H200 (140GB each)
- **Memory:** 700GB total
- **Storage:** Fast NVMe (1TB+)
- **Network:** High-speed interconnect

### For Training Edge Adapter Only (Recommended)
- **GPU:** 1-2× Nvidia A100 or RTX 6000
- **Memory:** 80-160GB
- **Storage:** SSD 500GB
- **Time:** 24-48 hours

### For Inference
- **GPU:** Jetson AGX Orin or RTX 6000 Ada
- **Compute:** ≥100 TFLOPS
- **Memory:** ≥8GB VRAM
- **I/O:** 1Gbps Ethernet for network

---

## Usage Examples

### Training
```bash
# Head-only training (recommended)
torchrun --nproc-per-node 2 vla-scripts/train_asyncvla.py \
  --vla_path ./AsyncVLA_release \
  --dataset_name asyncvla \
  --wandb_entity "your_entity" \
  --wandb_project "asyncvla" \
  --grad_accumulation_steps 2
```

### Inference
```bash
# Run inference on sample data
python inference/run_asyncvla.py

# Robot deployment
# See inference/run_asyncvla.py for ROS1 integration
```

---

## Innovation Highlights

1. **Asynchronous Decoupling**
   - Separates expensive computation from real-time control
   - Enables non-blocking network communication

2. **Edge-Optimized Architecture**
   - Lightweight multi-head attention (shead)
   - Parameter-efficient (5-10M params)

3. **Multi-Dataset Co-training**
   - Combines navigation, locomotion, social contexts
   - Improves generalization

4. **LoRA Fine-tuning**
   - Efficient adaptation (5-10M trainable vs 7B total)
   - Maintains pre-trained knowledge

5. **Robust Network Handling**
   - Graceful degradation on latency
   - Fallback to cached predictions

---

## Advantages vs Traditional VLA

| Aspect | Traditional VLA | AsyncVLA |
|--------|-----------------|----------|
| Latency | 200-500ms | 60-130ms |
| Speedup | 1× | 3-5× |
| Deployment | GPU server only | Edge + cloud |
| Scalability | Poor on robots | Good |
| Network robust | No | Yes |
| Training cost | High (H100/H200) | Lower (A100) |

---

## Limitations & Future Work

### Current Limitations
1. Requires network connectivity (WiFi/Ethernet)
2. Training edge adapter needs labeled trajectories
3. Assumes reliable GPS/compass (can degrade outdoors/indoors)
4. Limited to discrete action tokenization

### Future Directions
1. **Quantization** - INT8 for faster edge inference
2. **Temporal Modeling** - RNNs for sequential prediction
3. **Multi-modal Fusion** - LiDAR + RGB integration
4. **Adaptive Computation** - Dynamic depth networks

---

## Related Work

**Built on:**
- OpenVLA-OFT: Base VLA architecture
- OmniVLA: Multi-modal fusion
- ViNT: Visual navigation transformer

**Related Papers:**
- "Visual Navigation in Real-World Indoor Environments Using End-to-End Deep Learning" (Shah et al., 2022)
- "Open-Vocabulary Mobile Manipulation" (Kim et al., 2023)
- "Transformer-based Reinforcement Learning for Goal-directed Navigation" (Shah et al., 2021)

---

## Installation & Setup

```bash
# Create environment
conda create -n asyncvla python=3.10 -y
conda activate asyncvla

# Install PyTorch
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# Clone and install
git clone https://github.com/NHirose/AsyncVLA.git
cd AsyncVLA
pip install -e .

# Install Flash Attention 2 (for training)
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

# Download checkpoints
git clone https://huggingface.co/NHirose/AsyncVLA_release
```

---

## Testing & Validation

### Inference Test
```python
from inference.run_asyncvla import Inference

# Load models
inference = Inference(
    save_dir="./inference",
    lan_inst_prompt="move toward goal",
    goal_utm=utm_coords,
    goal_compass=compass_heading,
    goal_image_PIL=goal_image,
    action_tokenizer=tokenizer,
    processor=processor
)

# Run inference loop
inference.run_asyncvla()
```

---

## Performance Benchmarks

### Latency Comparison
- **End-to-end VLA:** 250-500ms
- **AsyncVLA (base):** 30-50ms (base only)
- **AsyncVLA (total):** 85-130ms (full pipeline)
- **Speedup:** 3-5×

### Accuracy
- Maintained comparable accuracy to end-to-end VLA
- Better generalization on unseen environments
- Robust to network jitter (5-30ms variations)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM during training | Reduce batch size or use gradient checkpointing |
| Slow inference | Check network latency, reduce image size |
| Poor navigation | Verify goal pose encoding, check GPS accuracy |
| Network timeouts | Increase timeout threshold, check WiFi signal |
| Model loading errors | Reinstall transformers and safetensors |

---

## Key Papers to Read

1. "OpenVLA: An Open-Source Vision-Language-Action Model for Robotic Manipulation" (Kim et al., 2024)
2. "Transformer-based Visuo-Motor Control for Real-World Manipulation" (Brohan et al., 2023)
3. "Learning to Navigate Unseen Environments: Back Translation with Environmental Dropout" (Shah et al., 2021)

---

## Contact & Resources

- **Project Page:** https://asyncvla.github.io/
- **GitHub:** https://github.com/NHirose/AsyncVLA
- **Checkpoints:** https://huggingface.co/NHirose/AsyncVLA_release
- **Paper:** https://arxiv.org/abs/2602.13476

---

## Summary

AsyncVLA represents a significant step toward practical deployment of Vision-Language-Action models on resource-constrained robots. By intelligently splitting computation between remote servers and edge devices, it achieves **3-5× latency reduction** while maintaining robustness through asynchronous communication. The framework is production-ready and opens new possibilities for real-time autonomous navigation.

**Key Takeaway:** Asynchronous decoupling enables powerful AI models to run on edge robots without sacrificing latency or accuracy.

