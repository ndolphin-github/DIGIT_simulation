# AsyncVLA: Code Implementation Reference

## Table of Contents
1. [Core Model Classes](#core-model-classes)
2. [Training Loop](#training-loop)
3. [Inference Pipeline](#inference-pipeline)
4. [Key Functions](#key-functions)
5. [Configuration Examples](#configuration-examples)

---

## Core Model Classes

### 1. OpenVLA Base Model

```python
class OpenVLA(PrismaticVLM):
    """
    Wraps OpenVLA with action tokenization capabilities.
    Inherits from PrismaticVLM (Vision-Language Model).
    """
    
    def __init__(
        self,
        *args,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
        action_tokenizer: ActionTokenizer,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm_stats = norm_stats
        self.action_tokenizer = action_tokenizer

    @torch.inference_mode()
    def predict_action(
        self,
        image: Image,
        instruction: str,
        unnorm_key: Optional[str] = None,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core inference function: Image + Instruction → Action
        
        Args:
            image: PIL Image (H, W, 3)
            instruction: Task description string
            unnorm_key: Dataset name for denormalization stats
            
        Returns:
            np.ndarray: Unnormalized continuous action [action_dim]
        """
        # 1. Get processor components
        image_transform = self.vision_backbone.image_transform
        tokenizer = self.llm_backbone.tokenizer
        
        # 2. Build prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(
            role="human",
            message=f"What action should the robot take to {instruction.lower()}?"
        )
        prompt_text = prompt_builder.get_prompt()
        
        # 3. Tokenize prompt
        input_ids = tokenizer(
            prompt_text,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        # 4. Handle LLaMA-specific empty token
        if isinstance(tokenizer, LlamaTokenizerFast):
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.tensor([[29871]]).to(input_ids.device)),
                    dim=1
                )
        
        # 5. Process image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {
                k: v[None, ...].to(self.device)
                for k, v in pixel_values.items()
            }
        
        # 6. Generate actions
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype):
            generated_ids = super(PrismaticVLM, self).generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=self.get_action_dim(unnorm_key),
                **kwargs
            )
        
        # 7. Extract and decode action tokens
        predicted_action_token_ids = generated_ids[
            0, -self.get_action_dim(unnorm_key):
        ]
        normalized_actions = (
            self.action_tokenizer.decode_token_ids_to_actions(
                predicted_action_token_ids.cpu().numpy()
            )
        )
        
        # 8. Denormalize actions
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        action_norm_stats = self.norm_stats[unnorm_key]["action"]
        
        mask = action_norm_stats.get(
            "mask",
            np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high = np.array(action_norm_stats["q99"])
        action_low = np.array(action_norm_stats["q01"])
        
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low)
            + action_low,
            normalized_actions,
        )
        
        return actions
```

### 2. Proprioceptive Projector

```python
class ProprioProjector(nn.Module):
    """
    Projects proprioceptive features (pose) to LLM embedding space.
    
    Input: Goal pose in robot frame (cos(θ), sin(θ), Δx, Δy)
    Output: Embedded representation matching LLM dimension
    """
    
    def __init__(self, llm_dim: int, proprio_dim: int = 4):
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim
        
        # Linear projection: 4D pose → llm_dim
        self.proj = nn.Linear(proprio_dim, llm_dim)
        self.ln = nn.LayerNorm(llm_dim)
    
    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proprio: [batch, 4] or [batch, 1, 4]
                     (cos θ, sin θ, Δx, Δy)
        
        Returns:
            [batch, 1, llm_dim] projected embeddings
        """
        if proprio.dim() == 2:
            proprio = proprio.unsqueeze(1)  # [batch, 1, 4]
        
        # Project to embedding space
        embedded = self.proj(proprio)  # [batch, 1, llm_dim]
        
        # Layer normalization
        embedded = self.ln(embedded)
        
        return embedded  # [batch, 1, llm_dim]
```

### 3. Action Tokenizer

```python
class ActionTokenizer:
    """
    Discretizes continuous actions into tokens and vice versa.
    
    Maps continuous action space [-1, 1] to discrete token IDs.
    """
    
    def __init__(self, tokenizer, num_actions_bins: int = 256):
        self.tokenizer = tokenizer
        self.num_actions_bins = num_actions_bins
        
        # Special action tokens
        self.action_start_token = "<action>"
        self.action_end_token = "</action>"
    
    def encode_actions(self, actions: np.ndarray) -> str:
        """
        Convert continuous actions [-1, 1] to token string.
        
        Args:
            actions: [num_actions, action_dim]
        
        Returns:
            str: Tokenized action sequence
        """
        tokens = []
        for action in actions:
            # Quantize each action dimension
            bin_indices = np.digitize(
                action,
                np.linspace(-1, 1, self.num_actions_bins)
            )
            bin_indices = np.clip(
                bin_indices,
                0,
                self.num_actions_bins - 1
            )
            tokens.extend(bin_indices.tolist())
        
        return self.action_start_token + ",".join(
            str(t) for t in tokens
        ) + self.action_end_token
    
    def decode_token_ids_to_actions(
        self,
        token_ids: np.ndarray
    ) -> np.ndarray:
        """
        Convert token IDs back to continuous actions.
        
        Args:
            token_ids: [num_actions, num_tokens_per_action]
        
        Returns:
            np.ndarray: [num_actions, action_dim] in [-1, 1]
        """
        actions = []
        bins = np.linspace(-1, 1, self.num_actions_bins)
        
        for token_id in token_ids:
            action_value = bins[int(token_id) % self.num_actions_bins]
            actions.append(action_value)
        
        return np.array(actions)
    
    def __call__(self, actions: np.ndarray) -> str:
        """Encode actions to string."""
        return self.encode_actions(actions)
```

### 4. Edge Adapter (shead)

```python
class Edge_adapter(nn.Module):
    """
    Lightweight multi-head attention for on-robot processing.
    
    Takes hidden states from base VLA and refines them with
    current sensor observations.
    """
    
    def __init__(
        self,
        obs_encoding_size: int = 1024,
        mha_num_attention_heads: int = 8,
        mha_num_attention_layers: int = 2,
        mha_ff_dim_factor: int = 4,
    ):
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=obs_encoding_size,
                num_heads=mha_num_attention_heads,
                batch_first=True,
                dropout=0.1,
            )
            for _ in range(mha_num_attention_layers)
        ])
        
        # Feed-forward networks (one per attention layer)
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    obs_encoding_size,
                    obs_encoding_size * mha_ff_dim_factor
                ),
                nn.GELU(),
                nn.Linear(
                    obs_encoding_size * mha_ff_dim_factor,
                    obs_encoding_size
                ),
            )
            for _ in range(mha_num_attention_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(obs_encoding_size)
            for _ in range(2 * mha_num_attention_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, obs_encoding_size]
            attn_mask: Optional attention mask
        
        Returns:
            [batch, seq_len, obs_encoding_size] refined representations
        """
        for i, (attn_layer, ff_layer) in enumerate(
            zip(self.attention_layers, self.ff_layers)
        ):
            # Multi-head self-attention
            x_norm = self.layer_norms[2 * i](x)
            attn_out, _ = attn_layer(
                x_norm,
                x_norm,
                x_norm,
                attn_mask=attn_mask,
            )
            x = x + attn_out  # Residual
            
            # Feed-forward
            x_norm = self.layer_norms[2 * i + 1](x)
            ff_out = ff_layer(x_norm)
            x = x + ff_out  # Residual
        
        return x
```

### 5. Action Decoding Heads

```python
class L1RegressionActionHead_idcat(nn.Module):
    """
    Predicts continuous actions via L1 regression.
    
    Input: Hidden states from VLA
    Output: Continuous actions in normalized space [-1, 1]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_actions_chunk: int = 8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions_chunk = num_actions_chunk
        
        # MLP for action prediction
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Output head for multiple action chunks
        self.head = nn.Linear(
            hidden_dim,
            num_actions_chunk * action_dim
        )
        
        # Action normalization
        self.register_buffer(
            'action_mean',
            torch.zeros(action_dim)
        )
        self.register_buffer(
            'action_std',
            torch.ones(action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        
        Returns:
            [batch, num_actions_chunk, action_dim] normalized actions
        """
        # Pool over sequence dimension (take last token)
        pooled = x[:, -1, :]  # [batch, input_dim]
        
        # Pass through MLP
        hidden = self.mlp(pooled)  # [batch, hidden_dim]
        
        # Predict actions
        actions_flat = self.head(hidden)  # [batch, num_actions * action_dim]
        
        # Reshape to action chunks
        actions = actions_flat.reshape(
            -1,
            self.num_actions_chunk,
            self.action_dim
        )  # [batch, num_actions, action_dim]
        
        # Normalize to [-1, 1]
        actions = torch.tanh(actions)
        
        return actions


class Proj_Actiontokens(nn.Module):
    """
    Projects hidden states to action token space.
    
    Used for token-based action prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.action_dim = action_dim
        
        # Projection layers
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        
        Returns:
            [batch, seq_len, action_dim] token logits
        """
        return self.proj(x)
```

---

## Training Loop

### Main Training Function

```python
def train_asyncvla(cfg: OmniVLAConfig) -> None:
    """
    Main training loop for AsyncVLA.
    Trains on multiple datasets concurrently.
    """
    
    # 1. Initialize distributed training
    distributed_state = Accelerator()
    device_id = distributed_state.device
    
    # 2. Load models
    processor = AutoProcessor.from_pretrained(
        cfg.vla_path,
        trust_remote_code=True
    )
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device_id)
    
    # 3. Initialize action tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    
    # 4. Setup LoRA fine-tuning
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all_linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
    
    # 5. Initialize components
    pose_projector = ProprioProjector(
        llm_dim=vla.llm_dim,
        proprio_dim=POSE_DIM
    )
    
    action_head = L1RegressionActionHead_idcat(
        input_dim=vla.llm_dim,
        hidden_dim=vla.llm_dim,
        action_dim=ACTION_DIM,
        num_actions_chunk=NUM_ACTIONS_CHUNK,
    )
    
    shead = Edge_adapter(
        obs_encoding_size=1024,
        mha_num_attention_heads=8,
        mha_num_attention_layers=2,
        mha_ff_dim_factor=4,
    )
    
    # 6. Create datasets
    train_datasets = [
        GNMDataset(config["gnm"]),
        LeLaNDataset(config["lelan"]),
        SACSoNDataset(config["sacson"]),
    ]
    
    # 7. Create dataloaders
    train_loaders = [
        DataLoader(
            ds,
            batch_size=cfg.batch_size,
            sampler=DistributedSampler(ds),
            collate_fn=PaddedCollatorForActionPrediction_Nav_MMN(
                cfg.tokenizer_max_length,
                cfg.pad_token_id,
            ),
        )
        for ds in train_datasets
    ]
    
    # 8. Setup optimizer
    trainable_params = [
        p for p in vla.parameters() if p.requires_grad
    ]
    trainable_params += list(action_head.parameters())
    trainable_params += list(shead.parameters())
    trainable_params += list(pose_projector.parameters())
    
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],
        gamma=0.1,
    )
    
    # 9. Training loop
    step = 0
    for epoch in range(100):
        for sampler in [s for s in train_loaders]:
            sampler.set_epoch(epoch)
        
        iters = [iter(loader) for loader in train_loaders]
        
        while step < cfg.max_steps:
            # Cycle through datasets
            for dataset_idx, iterator in enumerate(iters):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iters[dataset_idx] = iter(train_loaders[dataset_idx])
                    batch = next(iters[dataset_idx])
                
                # Forward pass
                loss, metrics = run_forward_pass(
                    vla=vla,
                    action_head=action_head,
                    shead=shead,
                    pose_projector=pose_projector,
                    batch=batch,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                )
                
                # Backward pass
                loss.backward()
                
                if (step + 1) % cfg.grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        max_norm=1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                
                step += 1
                
                # Logging
                if step % 100 == 0:
                    print(f"Step {step}: Loss = {loss:.4f}")
                
                if step >= cfg.max_steps:
                    break
    
    # 10. Save checkpoint
    torch.save(
        vla.state_dict(),
        f"{cfg.output_dir}/vla_final.pt"
    )
    torch.save(
        action_head.state_dict(),
        f"{cfg.output_dir}/action_head_final.pt"
    )
```

---

## Inference Pipeline

### Complete Inference Example

```python
def run_inference_asyncvla(
    vla,
    action_head,
    shead,
    pose_projector,
    processor,
    action_tokenizer,
    current_image: PIL.Image,
    goal_image: PIL.Image,
    goal_pose: np.ndarray,
    language_instruction: str,
    device_id: int,
) -> np.ndarray:
    """
    Complete inference pipeline.
    
    Args:
        current_image: Current camera frame [224, 224, 3]
        goal_image: Goal camera frame [224, 224, 3]
        goal_pose: [cos(θ), sin(θ), Δx, Δy]
        language_instruction: Task description
    
    Returns:
        np.ndarray: [8, 4] actions (Δx, Δy, Δθ, v)
    """
    
    # 1. Prepare inputs
    with torch.no_grad():
        # Image preprocessing
        img_transform = processor.image_processor.apply_transform
        pixel_values = torch.stack([
            img_transform(current_image),
            img_transform(goal_image),
        ]).to(device_id).to(torch.bfloat16)
        pixel_values = pixel_values.unsqueeze(0)  # [1, 2, 3, 224, 224]
        
        # Language tokenization
        prompt = (
            f"What action should the robot take to {language_instruction}?"
        )
        input_ids = processor.tokenizer(
            prompt,
            return_tensors="pt"
        ).input_ids.to(device_id)
        
        # Pose projection
        goal_pose_tensor = torch.tensor(
            goal_pose,
            dtype=torch.bfloat16,
            device=device_id
        ).unsqueeze(0)  # [1, 4]
        goal_pose_proj = pose_projector(goal_pose_tensor)  # [1, 1, llm_dim]
        
        # 2. Base VLA forward pass
        vla_output = vla(
            input_ids=input_ids,
            pixel_values=pixel_values,
            proprio=goal_pose_tensor,
            proprio_projector=pose_projector,
            output_hidden_states=True,
        )
        hidden_states = vla_output.hidden_states[-1]  # [1, seq_len, llm_dim]
        
        # 3. Edge adapter processing
        adapted_states = shead(hidden_states)  # [1, seq_len, llm_dim]
        
        # 4. Action head decoding
        normalized_actions = action_head(adapted_states)  # [1, 8, 4]
        normalized_actions = normalized_actions.cpu().numpy()[0]
        
        # 5. Denormalize actions
        actions = denormalize_actions(
            normalized_actions,
            q01=np.array([-2.0, -2.0, -np.pi, 0.0]),
            q99=np.array([2.0, 2.0, np.pi, 1.0]),
        )
    
    return actions
```

---

## Key Functions

### Action Denormalization

```python
def denormalize_actions(
    normalized: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
) -> np.ndarray:
    """
    Convert from normalized [-1, 1] to real-world values.
    
    Formula: a_real = 0.5 * (a_norm + 1) * (q99 - q01) + q01
    """
    return 0.5 * (normalized + 1) * (q99 - q01) + q01


def pose_encoding(
    current_pose: Tuple[float, float],
    goal_pose: Tuple[float, float],
    heading_rad: float,
) -> np.ndarray:
    """
    Encode relative pose for proprioceptive input.
    
    Returns: [cos(θ), sin(θ), Δx, Δy]
    """
    delta_x = goal_pose[0] - current_pose[0]
    delta_y = goal_pose[1] - current_pose[1]
    
    return np.array([
        np.cos(heading_rad),
        np.sin(heading_rad),
        delta_x,
        delta_y,
    ])
```

### PD Controller for Smooth Trajectory

```python
class PDController:
    """Proportional-Derivative controller for smooth robot motion."""
    
    def __init__(self, kp: float = 0.5, kd: float = 0.1):
        self.kp = kp
        self.kd = kd
        self.prev_error = None
    
    def compute(self, desired: np.ndarray, actual: np.ndarray) -> np.ndarray:
        """
        Compute control command.
        
        Args:
            desired: [Δx, Δy, Δθ, v]
            actual: Current state
        
        Returns:
            Control command
        """
        error = desired - actual
        
        # Proportional term
        p_term = self.kp * error
        
        # Derivative term
        if self.prev_error is not None:
            d_term = self.kd * (error - self.prev_error)
        else:
            d_term = np.zeros_like(error)
        
        self.prev_error = error
        
        return p_term + d_term
```

---

## Configuration Examples

### Training Configuration

```yaml
# config_nav/dataset_config.yaml
dataset_config_gnm:
  image: "./data/gnm/images"
  pickle: "./data/gnm/pickles"
  backside: false
  aug_seq:
    - random_crop
    - horizontal_flip
  only_front: true
  image_size: 224
  len_traj_pred: 8
  learn_angle: true
  context_size: 5
  context_type: "relative"
  normalize: true
  obs_encoding_size: 1024

dataset_config_lelan:
  image: "./data/lelan/images"
  pickle: "./data/lelan/pickles"
  backside: true
  image_size: 224
  len_traj_pred: 8

dataset_config_sacson:
  image: "./data/sacson/images"
  pickle: "./data/sacson/pickles"
  image_size: 224
  len_traj_pred: 8

# Edge adapter configuration
mha_num_attention_heads: 8
mha_num_attention_layers: 2
mha_ff_dim_factor: 4
```

### Dataclass Configuration

```python
@dataclass
class OmniVLAConfig:
    # Model
    vla_path: str = "./AsyncVLA_release"
    use_lora: bool = True
    lora_rank: int = 64
    lora_dropout: float = 0.05
    num_images_in_input: int = 2
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 2_000_000
    num_steps_before_decay: int = 750_000
    grad_accumulation_steps: int = 2
    
    # Data
    batch_size: int = 6
    tokenizer_max_length: int = 2048
    pad_token_id: int = 32000
    
    # Checkpointing
    output_dir: str = "./checkpoints"
    save_interval: int = 5000
    
    # Logging
    wandb_entity: str = "user"
    wandb_project: str = "asyncvla"
    log_interval: int = 100
```

---

## Error Handling & Best Practices

### Robust Inference with Error Handling

```python
def robust_inference(
    model,
    batch,
    device_id,
    max_retries: int = 3,
    timeout: float = 5.0,
) -> Optional[np.ndarray]:
    """
    Inference with error handling and fallback.
    """
    for attempt in range(max_retries):
        try:
            # Set timeout
            start_time = time.time()
            
            # Forward pass
            with torch.no_grad():
                output = model(batch)
            
            # Check timeout
            if time.time() - start_time > timeout:
                print(f"Warning: Inference exceeded timeout ({timeout}s)")
            
            return output.cpu().numpy()
        
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                # Clear cache and retry
                torch.cuda.empty_cache()
                print(f"Attempt {attempt + 1}: OOM, clearing cache and retrying")
                continue
            else:
                raise
        
        except Exception as e:
            print(f"Attempt {attempt + 1}: {type(e).__name__}: {e}")
            if attempt == max_retries - 1:
                return None  # Fallback
    
    return None
```

This comprehensive code reference covers the core implementation details of AsyncVLA!

