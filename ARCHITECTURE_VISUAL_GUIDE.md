# AsyncVLA Architecture: Detailed Visual Guide

## 1. System Architecture Overview

```
                        ┌─────────────────────────────────────────────┐
                        │      REMOTE WORKSTATION (GPU Server)        │
                        │                                             │
                        │  ┌───────────────────────────────────────┐  │
                        │  │     OpenVLA Base Model (7B)           │  │
                        │  │  ┌─────────────────────────────────┐  │  │
                        │  │  │  Vision Backbone (DINOv2)       │  │  │
                        │  │  │  • 224×224 image encoding       │  │  │
                        │  │  │  • ViT patch embedding          │  │  │
                        │  │  │  • 86M parameters               │  │  │
                        │  │  └─────────────────────────────────┘  │  │
                        │  │                 ↓                      │  │
                        │  │  ┌─────────────────────────────────┐  │  │
                        │  │  │  Multi-Modal Projector          │  │  │
                        │  │  │  • Fuses vision + text          │  │  │
                        │  │  │  • 50M parameters               │  │  │
                        │  │  └─────────────────────────────────┘  │  │
                        │  │                 ↓                      │  │
                        │  │  ┌─────────────────────────────────┐  │  │
                        │  │  │  Language Backbone (LLaMA-7B)   │  │  │
                        │  │  │  • Causal language modeling     │  │  │
                        │  │  │  • 7B parameters                │  │  │
                        │  │  │  • Output: [seq, 4096]          │  │  │
                        │  │  └─────────────────────────────────┘  │  │
                        │  │                 ↓                      │  │
                        │  │  ┌─────────────────────────────────┐  │  │
                        │  │  │  LoRA Adapters                  │  │  │
                        │  │  │  • Applied to all Linear layers │  │  │
                        │  │  │  • 5-10M trainable parameters   │  │  │
                        │  │  └─────────────────────────────────┘  │  │
                        │  │                 ↓                      │  │
                        │  │        Token Predictions (Seq)         │  │
                        │  └───────────────────────────────────────┘  │
                        │                                             │
                        └─────────────────────────────────────────────┘
                                          ↓
                        ┌─────────────────────────────────────────────┐
                        │         NETWORK COMMUNICATION               │
                        │  • ROS1 bridge for compatibility            │
                        │  • Asynchronous message passing             │
                        │  • Handles 5-20ms network latency           │
                        │  • Message format: token predictions        │
                        └─────────────────────────────────────────────┘
                                          ↓
                        ┌─────────────────────────────────────────────┐
                        │        ROBOT EDGE CONTROLLER                │
                        │    (Jetson AGX Orin / RTX 6000 Ada)         │
                        │                                             │
                        │  ┌───────────────────────────────────────┐  │
                        │  │     Edge Adapter (shead)              │  │
                        │  │  ┌─────────────────────────────────┐  │  │
                        │  │  │  Token Embedding Layer          │  │  │
                        │  │  │  • [seq, 4096] input            │  │  │
                        │  │  └─────────────────────────────────┘  │  │
                        │  │                 ↓                      │  │
                        │  │  ┌─────────────────────────────────┐  │  │
                        │  │  │  Multi-Head Attention           │  │  │
                        │  │  │  (2 layers, 8 heads)            │  │  │
                        │  │  │  • Self-attention over sequence │  │  │
                        │  │  │  • Query, Key, Value projection │  │  │
                        │  │  │  • [seq, 4096] → [seq, 4096]    │  │  │
                        │  │  └─────────────────────────────────┘  │  │
                        │  │                 ↓                      │  │
                        │  │  ┌─────────────────────────────────┐  │  │
                        │  │  │  Proprioceptive Projector       │  │  │
                        │  │  │  • Goal pose conditioning       │  │  │
                        │  │  │  • [cos(θ), sin(θ), Δx, Δy]    │  │  │
                        │  │  │  • Maps to [1, 4096]            │  │  │
                        │  │  └─────────────────────────────────┘  │  │
                        │  │                 ↓                      │  │
                        │  │  ┌─────────────────────────────────┐  │  │
                        │  │  │  Action Decoding Head           │  │  │
                        │  │  │  ┌──────────────────────────┐    │  │
                        │  │  │  │ L1RegressionActionHead   │    │  │
                        │  │  │  │ • [seq, 4096] → [8, 4]   │    │  │
                        │  │  │  │ • Direct action regression│    │  │
                        │  │  │  │ • Output: Δx, Δy, Δθ, v  │    │  │
                        │  │  │  └──────────────────────────┘    │  │
                        │  │  │  OR                               │  │
                        │  │  │  ┌──────────────────────────┐    │  │
                        │  │  │  │ Proj_ActionTokens        │    │  │
                        │  │  │  │ • [seq, 4096] → [1024]   │    │  │
                        │  │  │  │ • Token-based prediction  │    │  │
                        │  │  │  └──────────────────────────┘    │  │
                        │  │  └─────────────────────────────────┘  │  │
                        │  │                 ↓                      │  │
                        │  │        Action Outputs [8, 4]           │  │
                        │  │        (Normalized: [-1, 1])           │  │
                        │  └───────────────────────────────────────┘  │
                        │                                             │
                        │  ┌───────────────────────────────────────┐  │
                        │  │     Action Denormalization             │  │
                        │  │  • Reverse normalization               │  │
                        │  │  • Apply percentile bounds (q01, q99)  │  │
                        │  │  • Output: Real-world values           │  │
                        │  └───────────────────────────────────────┘  │
                        │                                             │
                        │  ┌───────────────────────────────────────┐  │
                        │  │     PD Controller                      │  │
                        │  │  • Smooth trajectory generation        │  │
                        │  │  • Low-pass filtering                  │  │
                        │  │  • Motor command generation            │  │
                        │  └───────────────────────────────────────┘  │
                        │                                             │
                        └─────────────────────────────────────────────┘
                                          ↓
                        ┌─────────────────────────────────────────────┐
                        │        ROBOT ACTUATORS & SENSORS            │
                        │  • Motor control (PWM signals)              │
                        │  • IMU/Compass feedback                     │
                        │  • GPS/Wheel odometry                       │
                        │  • Camera frame capture                     │
                        └─────────────────────────────────────────────┘
```

---

## 2. Data Flow During Inference

```
INPUT PROCESSING
════════════════════════════════════════════════════════════════════════════════

TIME = 0ms
─────────────────────────────────────────────────────────────────────────────
┌─────────────────────────┐
│  Current Camera Frame   │  ────► Image Preprocessing ────► [1, 3, 224, 224]
│  (RGB 224×224)          │            ↓
└─────────────────────────┘         • Resize to 224×224
                                    • Normalize (mean/std)
                                    
┌─────────────────────────┐
│  Goal Image             │  ────► Same preprocessing
│  (RGB 224×224)          │
└─────────────────────────┘
                
┌─────────────────────────┐
│  Language Instruction   │  ────► Tokenization ────► [1, seq_len]
│  "move to blue bin"     │         ↓
└─────────────────────────┘       • BPE tokenization
                                  • Padding/truncation
                                  
┌─────────────────────────┐
│  Robot Pose             │  ────► Pose Encoding ────► [1, 1, 4]
│  (GPS, compass)         │         ↓
└─────────────────────────┘       • [cos(θ), sin(θ), Δx, Δy]
                                  • Normalization


BASE VLA PROCESSING (Remote Workstation)
════════════════════════════════════════════════════════════════════════════════

TIME = 5-30ms
─────────────────────────────────────────────────────────────────────────────

INPUT CONCATENATION:
  ┌──────────────────────────────────────┐
  │ [input_ids]         [pixel_values]   │
  │ [seq_len]           [1, 3, 224, 224] │
  │ Token IDs           Image patches    │
  └──────────────────────────────────────┘
           ↓                    ↓
    ┌────────────────────────────────────┐
    │      Vision Encoder (DINOv2)       │
    │  224×224 → 14×14 patches (196)     │
    │  [14×14, 768] patch embeddings     │
    └────────────────────────────────────┘
           ↓
    ┌────────────────────────────────────┐
    │   Multi-Modal Projector            │
    │  [196, 768] → [196, 4096]          │
    │  Aligns vision to LLM dimension    │
    └────────────────────────────────────┘
           ↓
    ┌────────────────────────────────────┐
    │  Concatenate with text tokens      │
    │  [vision: 196, 4096]               │
    │  [text: seq_len, 4096]             │
    │  [pose: 1, 4096] (via projector)   │
    │  ─────────────────────────────     │
    │  Total: [seq_total, 4096]          │
    └────────────────────────────────────┘
           ↓
    ┌────────────────────────────────────┐
    │    Language Model (LLaMA-7B)       │
    │    ┌──────────────────────────┐    │
    │    │ Transformer Decoder      │    │
    │    │ • 32 layers              │    │
    │    │ • 32 attention heads      │    │
    │    │ • 4096 hidden dim         │    │
    │    │ • Causal masking          │    │
    │    └──────────────────────────┘    │
    │    Input: [seq_total, 4096]        │
    │    Output: [seq_total, 4096]       │
    │    (+ LoRA adapters applied)       │
    └────────────────────────────────────┘
           ↓
    ┌────────────────────────────────────┐
    │    Action Token Prediction Head    │
    │    Linear([4096] → [vocab_size])   │
    │    ───────────────────────────────  │
    │    Output: Token logits for        │
    │    8-action sequence               │
    │    [8, vocab_size]                 │
    └────────────────────────────────────┘
           ↓
    ┌────────────────────────────────────┐
    │    Sampling / Decoding             │
    │    • Greedy or temperature sampling│
    │    • Token ID → Action values      │
    │    Output: [8, 4] action tokens    │
    └────────────────────────────────────┘


NETWORK TRANSMISSION
════════════════════════════════════════════════════════════════════════════════

TIME = 30-50ms
─────────────────────────────────────────────────────────────────────────────

  Base VLA Result (token predictions)
           ↓
  ┌──────────────────────────┐
  │  ROS Message Serialization│
  │  • Pack [8, 4] float32    │
  │  • Add timestamp          │
  │  • Checksums              │
  └──────────────────────────┘
           ↓
  ┌──────────────────────────┐
  │  UDP/TCP Transmission     │
  │  • Latency: 5-30ms        │
  │  • Network jitter handling│
  └──────────────────────────┘


EDGE PROCESSING (Robot Onboard)
════════════════════════════════════════════════════════════════════════════════

TIME = 50-100ms (Parallel with Base VLA)
─────────────────────────────────────────────────────────────────────────────

  WHILE Base VLA is computing, Edge is:
  
  1. Current Observation Encoding
     ┌──────────────────────────┐
     │ Current Camera [224×224]  │
     │ Past Camera [96×96]       │
     │ Recent IMU readings       │
     │ Encode to [obs_sz, 1024]  │
     └──────────────────────────┘
             ↓
  2. Proprioceptive Fusion
     ┌──────────────────────────┐
     │ Goal pose encoding        │
     │ [cos(θ), sin(θ), Δx, Δy] │
     │ Project to [1, 4096]      │
     └──────────────────────────┘
             ↓
  3. Receive Base VLA tokens
     ┌──────────────────────────┐
     │ Token predictions [8, 4]  │
     │ Convert to embeddings     │
     │ [8, 4] → [8, 4096]        │
     └──────────────────────────┘
             ↓
  4. Multi-Head Attention (shead)
     ┌──────────────────────────┐
     │ Layer 1:                  │
     │ • Query: [seq, 4096]      │
     │ • Key/Value: [seq, 4096]  │
     │ • Attention: [seq, 4096]  │
     │ • LayerNorm + FFN         │
     │                           │
     │ Layer 2:                  │
     │ • Same as Layer 1         │
     │ • Output: [seq, 4096]     │
     └──────────────────────────┘
             ↓
  5. Action Head Decoding
     ┌──────────────────────────┐
     │ L1Regression Head:        │
     │ [seq, 4096] → [8, 4]      │
     │ Denormalize to real units:│
     │ • Δx: [-2m, 2m]           │
     │ • Δy: [-2m, 2m]           │
     │ • Δθ: [-π, π]             │
     │ • v: [0, 1] m/s           │
     └──────────────────────────┘
             ↓
  6. Trajectory Smoothing (PD Controller)
     ┌──────────────────────────┐
     │ Apply low-pass filter     │
     │ Proportional-Derivative   │
     │ Control for smooth motion │
     │ Output: Motor commands    │
     └──────────────────────────┘
             ↓
  7. Execute Action
     ┌──────────────────────────┐
     │ Send PWM to motors        │
     │ Update odometry           │
     │ Read new sensors          │
     └──────────────────────────┘


LOOP TIMING
════════════════════════════════════════════════════════════════════════════════

Frame N Timeline:
  t=0ms:    Capture image N, send to base VLA
  t=5ms:    Edge processes image N with previous predictions
  t=25ms:   Base VLA finishes computing (still processing N)
  t=50ms:   Network transmission complete
  t=80ms:   Edge receives base VLA tokens for image N
  t=85ms:   Edge attention + action head completes
  t=100ms:  Execute action (8-step trajectory)
  
Frame N+1 Timeline (100ms later):
  t=100ms:  Capture image N+1, send to base VLA
  t=200ms:  Execute next action (from image N+1 prediction)
```

---

## 3. Training Pipeline Data Flow

```
                    ┌─────────────────────────────────────┐
                    │     MULTI-DATASET LOADING           │
                    └─────────────────────────────────────┘
                           ↓
         ┌─────────────────┼─────────────────┐
         ↓                 ↓                 ↓
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │   GNM       │  │   LeLaN     │  │  SACSoN     │
    │ (Goal Nav)  │  │ (Legged)    │  │ (Social)    │
    │             │  │             │  │             │
    │ Batch: 6    │  │ Batch: 6    │  │ Batch: 6    │
    │ Shape:      │  │ Shape:      │  │ Shape:      │
    │ [6,3,224]   │  │ [6,3,224]   │  │ [6,3,224]   │
    └─────────────┘  └─────────────┘  └─────────────┘
         ↓                 ↓                 ↓
         └─────────────────┼─────────────────┘
                           ↓
         ┌─────────────────────────────────────┐
         │      BATCH COLLATION & PADDING      │
         │  • Stack into [18, 3, 224, 224]     │
         │  • Add attention masks               │
         │  • Pad sequences                    │
         └─────────────────────────────────────┘
                           ↓
         ┌─────────────────────────────────────┐
         │   FORWARD PASS (Distributed)        │
         │  • GPU 0-4 with DDP                 │
         │  • Gradient accumulation (2 steps)  │
         └─────────────────────────────────────┘
                           ↓
         ┌─────────────────────────────────────┐
         │      BASE VLA (if TRAIN_BASE)       │
         │  ┌─────────────────────────────┐    │
         │  │ Vision Backbone + LLM       │    │
         │  │ Output: [seq, 4096]         │    │
         │  │ (LoRA enabled)              │    │
         │  └─────────────────────────────┘    │
         │              ↓                       │
         │  ┌─────────────────────────────┐    │
         │  │ Hidden State Extraction     │    │
         │  │ [batch, seq, 4096]          │    │
         │  └─────────────────────────────┘    │
         └─────────────────────────────────────┘
                           ↓
         ┌─────────────────────────────────────┐
         │    ACTION HEAD PROCESSING           │
         │  ┌─────────────────────────────┐    │
         │  │ L1RegressionActionHead      │    │
         │  │ [batch, seq, 4096]          │    │
         │  │         ↓                   │    │
         │  │ [batch, 8, 4] predictions   │    │
         │  └─────────────────────────────┘    │
         │           OR                        │
         │  ┌─────────────────────────────┐    │
         │  │ Proj_ActionTokens           │    │
         │  │ [batch, seq, 4096]          │    │
         │  │         ↓                   │    │
         │  │ [batch, seq, 1024] tokens   │    │
         │  └─────────────────────────────┘    │
         └─────────────────────────────────────┘
                           ↓
         ┌─────────────────────────────────────┐
         │   EDGE ADAPTER (shead) PROCESSING   │
         │  ┌─────────────────────────────┐    │
         │  │ Multi-Head Attention        │    │
         │  │ (2 layers, 8 heads)         │    │
         │  │ [batch, seq, 4096]          │    │
         │  │         ↓                   │    │
         │  │ [batch, seq, 4096] refined  │    │
         │  └─────────────────────────────┘    │
         │              ↓                       │
         │  ┌─────────────────────────────┐    │
         │  │ Action Decoding             │    │
         │  │ [batch, seq, 4096]          │    │
         │  │         ↓                   │    │
         │  │ [batch, 8, 4] actions       │    │
         │  └─────────────────────────────┘    │
         └─────────────────────────────────────┘
                           ↓
         ┌─────────────────────────────────────┐
         │      LOSS COMPUTATION               │
         │  ┌─────────────────────────────┐    │
         │  │ L2_actions Loss             │    │
         │  │ L2_pose_img Loss            │    │
         │  │ L2_img Loss                 │    │
         │  │ L2_lan Loss                 │    │
         │  │ L2_lan_pose Loss            │    │
         │  │ Total = Σ (w_i * L_i)       │    │
         │  └─────────────────────────────┘    │
         └─────────────────────────────────────┘
                           ↓
         ┌─────────────────────────────────────┐
         │    BACKWARD PASS & OPTIMIZATION     │
         │  • Gradient Accumulation (2 steps)  │
         │  • Multi-GPU communication (DDP)    │
         │  • Adam optimizer update            │
         │  • Learning rate decay              │
         └─────────────────────────────────────┘
                           ↓
         ┌─────────────────────────────────────┐
         │      LOGGING & CHECKPOINTING        │
         │  • Loss metrics to W&B              │
         │  • Model checkpoint every N steps   │
         │  • Validation metrics               │
         └─────────────────────────────────────┘
```

---

## 4. Key Dimensions Reference

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    TENSOR DIMENSION MAPPING                              ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║ INPUT TENSORS:                                                           ║
║ ──────────────────────────────────────────────────────────────────────   ║
║  • input_ids:        [batch, seq_len]                                   ║
║  • pixel_values:     [batch, 2, 3, 224, 224]  (2 images)               ║
║  • attention_mask:   [batch, seq_len]                                   ║
║  • goal_pose:        [batch, 4]  (cos θ, sin θ, Δx, Δy)                ║
║                                                                           ║
║ BASE VLA PROCESSING:                                                    ║
║ ──────────────────────────────────────────────────────────────────────   ║
║  • Vision patches:   [batch, num_patches=392, 768]                     ║
║    • 2 images × 196 patches/image = 392 patches                        ║
║    • Each patch: 768-dim (DINOv2)                                       ║
║                                                                           ║
║  • Projected vision: [batch, 392, 4096]                                ║
║                                                                           ║
║  • Goal pose proj:   [batch, 1, 4096]                                  ║
║                                                                           ║
║  • Text embeddings:  [batch, seq_len, 4096]                            ║
║                                                                           ║
║  • Concatenated:     [batch, 393+seq_len, 4096]                        ║
║                      (392 patches + 1 goal + seq tokens)                ║
║                                                                           ║
║  • LLM output:       [batch, 393+seq_len, 4096]                        ║
║                                                                           ║
║ ACTION PREDICTION:                                                      ║
║ ──────────────────────────────────────────────────────────────────────   ║
║  • Action head:      [batch, 8, 4]                                     ║
║    • 8 action steps  • 4 dimensions (Δx, Δy, Δθ, v)                    ║
║                                                                           ║
║  OR                                                                      ║
║                                                                           ║
║  • Token projection: [batch, seq_len, 1024]                            ║
║                      (tokenized action space)                           ║
║                                                                           ║
║ EDGE ADAPTER:                                                           ║
║ ──────────────────────────────────────────────────────────────────────   ║
║  • Input:            [batch, seq_len, 4096]                            ║
║  • Attention Layer 1:                                                   ║
║    - Query, Key, Value each [batch, seq_len, 4096]                     ║
║    - Attention scores: [batch, 8_heads, seq_len, seq_len]              ║
║    - Output: [batch, seq_len, 4096]                                     ║
║  • FFN: [batch, seq_len, 4096] → [batch, seq_len, 16384] → [batch, seq_len, 4096]  ║
║  • Layer 2: Repeat of Layer 1                                           ║
║  • Final output: [batch, seq_len, 4096]                                ║
║                                                                           ║
║ DENORMALIZATION:                                                        ║
║ ──────────────────────────────────────────────────────────────────────   ║
║  • Normalized: [-1, 1]                                                  ║
║  • Real-world:                                                          ║
║    - Δx: [-2.0, 2.0] meters                                            ║
║    - Δy: [-2.0, 2.0] meters                                            ║
║    - Δθ: [-π, π] radians                                               ║
║    - v:  [0, 1] m/s                                                    ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## 5. Training Configuration Matrix

```
╔════════════════════════════════════════════════════════════════════════════╗
║              TRAINING CONFIGURATION & HYPERPARAMETERS                     ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║ SCENARIO 1: FULL TRAINING (TRAIN_BASE=True, TRAIN_HEAD=True)             ║
║ ────────────────────────────────────────────────────────────────────────  ║
║  • Base VLA:         LoRA adapters + trainable (requires H100/H200)      ║
║  • Edge Adapter:     Fully trainable                                     ║
║  • Action Head:      Fully trainable                                     ║
║  • Total params:     ~7B + 5-10M trainable                              ║
║  • Requires:         5× Nvidia H200 GPUs (700GB total)                  ║
║  • Best for:         Custom task fine-tuning                            ║
║                                                                            ║
║ SCENARIO 2: HEAD-ONLY TRAINING (TRAIN_BASE=False, TRAIN_HEAD=True)      ║
║ ────────────────────────────────────────────────────────────────────────  ║
║  • Base VLA:         Frozen (inference only)                            ║
║  • Edge Adapter:     Fully trainable                                    ║
║  • Action Head:      Fully trainable                                    ║
║  • Total params:     5-15M trainable                                    ║
║  • Requires:         1-2× Nvidia A100 or RTX 6000 (80-160GB total)     ║
║  • Training time:    ~24-48 hours                                        ║
║  • Best for:         Most use cases (efficient + effective)             ║
║                                                                            ║
║ SCENARIO 3: INFERENCE-ONLY (No training)                                ║
║ ────────────────────────────────────────────────────────────────────────  ║
║  • All models:       Frozen                                             ║
║  • Requires:         1× RTX 6000 Ada or Jetson AGX Orin                ║
║  • Memory:           ~30GB (16GB VLA + 2GB edge + overhead)            ║
║  • Latency:          60-130ms total (30-50ms base + 10-30ms edge)       ║
║                                                                            ║
║ OPTIMIZATION SETTINGS:                                                   ║
║ ────────────────────────────────────────────────────────────────────────  ║
║  • Optimizer:        AdamW                                              ║
║  • Learning Rate:    1e-4 (linear warmup)                               ║
║  • Weight Decay:     0.01                                               ║
║  • Batch Size:       6 (GNM) + 6 (LeLaN) + 6 (SACSoN) = 18 total       ║
║  • Grad Accumulation:2 steps → effective batch = 36                     ║
║  • LR Scheduler:     MultiStepLR                                        ║
║    - Decay at step 750,000                                              ║
║    - Decay factor: 0.1                                                  ║
║  • Max Steps:        2,000,000 (nominal)                                ║
║  • Gradient Clip:    1.0                                                ║
║                                                                            ║
║ LoRA CONFIGURATION:                                                      ║
║ ────────────────────────────────────────────────────────────────────────  ║
║  • LoRA Rank (r):    64                                                 ║
║  • LoRA Alpha:       64                                                 ║
║  • LoRA Dropout:     0.05                                               ║
║  • Target Modules:   All linear layers (Transformers)                   ║
║  • Init Weights:     Gaussian                                           ║
║                                                                            ║
║ MIXED PRECISION:                                                         ║
║ ────────────────────────────────────────────────────────────────────────  ║
║  • Dtype:            bfloat16 (brain float 16)                          ║
║  • Advantages:       • Faster training                                   ║
║                      • Reduced memory usage                             ║
║                      • Better convergence than FP16                     ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
```

---

## 6. Real-time Execution Timeline

```
FULL EXECUTION CYCLE (with margins):
═════════════════════════════════════════════════════════════════════════════

Frame Capture Loop (100ms cycle):
┌─────────────────────────────────────────────────────────────────────────────┐
│ TIME  │ COMPONENT              │ OPERATION                  │ LATENCY       │
├──────┼────────────────────────┼────────────────────────────┼───────────────┤
│ 0ms  │ Robot Sensor Input     │ • Camera capture           │ 5-10ms        │
│      │ (Edge)                 │ • IMU/Compass read         │               │
│      │                        │ • GPS position update      │               │
├──────┼────────────────────────┼────────────────────────────┼───────────────┤
│ 5ms  │ Preprocessing          │ • Image resize (96×96)     │ 2-5ms         │
│ +2ms │ (Edge)                 │ • Normalization            │               │
│      │                        │ • Stack with past image    │               │
├──────┼────────────────────────┼────────────────────────────┼───────────────┤
│ 7ms  │ Network Transmission   │ • Serialize batch          │ 5-20ms        │
│ +10ms│ (WiFi/Ethernet)        │ • UDP/TCP send             │               │
├──────┼────────────────────────┼────────────────────────────┼───────────────┤
│ 17ms │ Base VLA Processing    │ • Vision encoding          │ 30-50ms       │
│ +30ms│ (Remote GPU)           │ • LLM forward pass         │               │
│      │                        │ • Action tokenization      │               │
├──────┼────────────────────────┼────────────────────────────┼───────────────┤
│ 47ms │ Network Transmission   │ • Receive tokens           │ 5-20ms        │
│ +10ms│ (WiFi/Ethernet)        │ • Deserialize              │               │
├──────┼────────────────────────┼────────────────────────────┼───────────────┤
│ 57ms │ Edge Adapter           │ • Multi-head attention     │ 10-15ms       │
│ +15ms│ (Robot Edge Device)    │ • Action decoding          │               │
│      │                        │ • Denormalization          │               │
├──────┼────────────────────────┼────────────────────────────┼───────────────┤
│ 72ms │ Post-processing        │ • PD controller filtering  │ 5ms           │
│ +5ms │ (Edge)                 │ • Motor command generation │               │
├──────┼────────────────────────┼────────────────────────────┼───────────────┤
│ 77ms │ Motor Control          │ • PWM signal dispatch      │ 5-10ms        │
│ +8ms │ (Edge)                 │ • Hardware interface       │               │
├──────┼────────────────────────┼────────────────────────────┼───────────────┤
│ 85ms │ READY FOR NEXT FRAME   │ (15ms buffer before 100ms) │ TOTAL: 85ms   │
└─────────────────────────────────────────────────────────────────────────────┘

CONCURRENT PROCESSING (Key Advantage of AsyncVLA):
═════════════════════════════════════════════════════════════════════════════

Frame N:
   0ms  ┌─ Capture image N
   5ms  │  └─ Send to Base VLA
   7ms  ├─ Base VLA starts processing (parallel work!)
   7ms  └─ Edge starts processing image N-1 predictions
  10ms     └─ Edge finishes early action
  37ms     ┌─ Base VLA completes
  47ms     │  └─ Network transmission
  57ms     ├─ Edge receives tokens N
  57ms     └─ Execute action (from old prediction)
  72ms        └─ New action ready
  100ms       └─ NEXT FRAME CAPTURE (Frame N+1)

Without AsyncVLA (Sequential):
   0ms  ┌─ Capture image N
   5ms  │  └─ Send to Base VLA
   7ms  │  └─ Base VLA processing...
  37ms  ├─ Base VLA completes
  47ms  │  └─ Network transmission
  57ms  ├─ Edge processes (can't start earlier!)
  72ms  │  └─ Action complete
  72ms  └─ Execute (ONLY NOW)
 100ms  ┌─ Next frame capture (action executed for only 28ms!)
        └─ TOO LATE FOR RESPONSIVE CONTROL

RESULT: AsyncVLA achieves 2-3× lower effective latency!
```

---

## 7. Memory Layout During Inference

```
GPU MEMORY USAGE (Robot Edge Device):
═════════════════════════════════════════════════════════════════════════════

Total Available: 24GB (RTX 6000 Ada)
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│ ┌────────────────────────────────────────────────────────────────────┐    │
│ │ Base VLA Model Weights (frozen)              [~14-16GB]            │    │
│ │ • Vision backbone (DINOv2)                   [~1GB]                │    │
│ │ • LLM (LLaMA-7B)                             [~13GB]               │    │
│ └────────────────────────────────────────────────────────────────────┘    │
│                                                                            │
│ ┌────────────────────────────────────────────────────────────────────┐    │
│ │ Edge Adapter + Action Heads (frozen)         [~0.05-0.1GB]         │    │
│ │ • Multi-head attention layers                [~0.03GB]             │    │
│ │ • Action decoding heads                      [~0.02GB]             │    │
│ │ • Proprioceptive projector                   [~0.01GB]             │    │
│ └────────────────────────────────────────────────────────────────────┘    │
│                                                                            │
│ ┌────────────────────────────────────────────────────────────────────┐    │
│ │ Cached KV Values (for fast inference)        [~1-2GB]              │    │
│ │ • Attention caches from previous tokens      [~1-2GB]              │    │
│ └────────────────────────────────────────────────────────────────────┘    │
│                                                                            │
│ ┌────────────────────────────────────────────────────────────────────┐    │
│ │ Activation Memory (runtime)                  [~0.5-1GB]            │    │
│ │ • Input tensors [batch=1, seq=393+seq_len]   [~0.1GB]             │    │
│ │ • Intermediate activations                   [~0.2GB]             │    │
│ │ • Output tensors [batch, 8, 4]               [~0.01GB]            │    │
│ │ • Temporary buffers                          [~0.2GB]             │    │
│ └────────────────────────────────────────────────────────────────────┘    │
│                                                                            │
│ ┌────────────────────────────────────────────────────────────────────┐    │
│ │ PyTorch Framework Overhead                   [~1-2GB]              │    │
│ │ • CUDA memory pools                          [~0.5GB]             │    │
│ │ • Autograd buffers (if training)             [~0GB in inference]  │    │
│ │ • Miscellaneous                              [~0.5GB]             │    │
│ └────────────────────────────────────────────────────────────────────┘    │
│                                                                            │
│ ┌────────────────────────────────────────────────────────────────────┐    │
│ │ HEADROOM / BUFFER                            [~5-6GB FREE]         │    │
│ └────────────────────────────────────────────────────────────────────┘    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

MEMORY PEAK TIMELINE:
  • Model Load: 16GB
  • Add activations: +1-2GB → 17-18GB
  • Output generation: ~18-19GB (peak)
  • Post-processing: 17-18GB
```

---

## 8. Error Handling & Robustness

```
ASYNCHRONOUS ERROR SCENARIOS:
═════════════════════════════════════════════════════════════════════════════

Scenario 1: Network Timeout
┌─────────────────────────────────────────────────────────────────────────────┐
│ • Base VLA sends tokens                                                     │
│ • Network packet lost (1-5% typical packet loss)                            │
│ • Edge timeout triggers (default: 500ms)                                    │
│ • Action: Use previous valid prediction or fallback to safety behavior      │
│ • Impact: Single frame delay (~100ms)                                       │
└─────────────────────────────────────────────────────────────────────────────┘

Scenario 2: Base VLA Computation Exceeds Expected Time
┌─────────────────────────────────────────────────────────────────────────────┐
│ • Base VLA takes 100ms instead of expected 50ms                             │
│ • Edge detects late arrival                                                 │
│ • Action: Edge uses most recent cached prediction                           │
│ • Result: Graceful degradation (latency increases slightly)                 │
└─────────────────────────────────────────────────────────────────────────────┘

Scenario 3: Edge Device Overload
┌─────────────────────────────────────────────────────────────────────────────┐
│ • Edge processing takes 20ms instead of 10ms                                │
│ • Queue builds up                                                           │
│ • Action: Skip old frames, process only latest                             │
│ • Result: Reduced latency by dropping intermediate predictions               │
└─────────────────────────────────────────────────────────────────────────────┘

Scenario 4: Sensor Data Unavailable
┌─────────────────────────────────────────────────────────────────────────────┐
│ • GPS fails (blocked indoors)                                               │
│ • Action: Use last valid GPS or relative dead reckoning                     │
│ • Impact: Pose drift, recovers when GPS returns                             │
└─────────────────────────────────────────────────────────────────────────────┘

BUFFER STRATEGY:
├─ Prediction Queue: [oldest_pred, current_pred, latest_pred]
├─ Drop Strategy: Always use most recent valid prediction
├─ Fallback: Last safe configuration (maintain previous action)
└─ Recovery: Re-synchronize when connection restored
```

This comprehensive visual guide shows the complete architecture and data flow of AsyncVLA!

