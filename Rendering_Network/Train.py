
import os
import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import our data loading module
from data_loader import DataConfig, create_dataloaders


# =========================
# Training Config
# =========================
@dataclass
class TrainingConfig:
    # Data config
    data_cfg: DataConfig = None
    
    # Training hyperparameters
    batch_size: int = 8
    epochs: int = 50
    lr: float = 2e-4
    weight_bg: float = 0.1                            # background loss weight
    bg_neutral_weight: float = 0.02                   # push bg toward 0 (i.e., 128 after de-centering)
    num_workers: int = 4
    save_every: int = 5                               # save sample preds every N epochs
    out_dir: str = r"e:\DIGIT_simulation\Rendering_Networks\TrainingOutput"
    
    def __post_init__(self):
        if self.data_cfg is None:
            self.data_cfg = DataConfig()


# =========================
# Model (U-Net Architecture)
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )
    def forward(self, x): return self.block(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 3, base: int = 48):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)   # 75x100 -> 37x50
        self.down2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)   # 37x50 -> 18x25
        self.bott  = DoubleConv(base*2, base*4)
        self.up2   = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)  # 18x25 -> 36x50
        self.conv2 = DoubleConv(base*4, base*2)
        self.up1   = nn.ConvTranspose2d(base*2, base, 2, stride=2)    # 36x50 -> 72x100
        # we'll pad/crop to 75x100 at the end
        self.conv1 = DoubleConv(base*2, base)
        self.head  = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        b  = self.bott(p2)
        u2 = self.up2(b)
        # align shapes (may differ by 1 row/col due to pooling rounding)
        if u2.size(-2) != d2.size(-2) or u2.size(-1) != d2.size(-1):
            u2 = F.interpolate(u2, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        c2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2)
        if u1.size(-2) != d1.size(-2) or u1.size(-1) != d1.size(-1):
            u1 = F.interpolate(u1, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        c1 = self.conv1(torch.cat([u1, d1], dim=1))
        y  = self.head(c1)
        # output in [-1,1] via tanh
        return torch.tanh(y)


# =========================
# Loss Function
# =========================
def masked_l1(pred, target, mask, weight_bg=0.1, bg_neutral_weight=0.02):
    """
    Masked L1 loss with background weighting.
    
    Args:
        pred: N x 3 x H x W in [-1,1] - predicted delta images
        target: N x 3 x H x W in [-1,1] - ground truth delta images
        mask: N x 1 x H x W in {0,1} - contact masks
        weight_bg: Weight for background loss
        bg_neutral_weight: Weight for encouraging neutral background
    """
    l1 = torch.abs(pred - target)  # N x 3 x H x W
    w  = mask                      # N x 1 x H x W

    num_c = w.sum() * pred.size(1) + 1e-8
    num_b = (1 - w).sum() * pred.size(1) + 1e-8

    loss_contact = (w * l1).sum() / num_c
    loss_bg      = ((1 - w) * l1).sum() / num_b

    # encourage near-zero background (0 ↔ Δ=128)
    bg_neutral = torch.abs((1 - w) * pred).mean()

    return loss_contact + weight_bg * loss_bg + bg_neutral_weight * bg_neutral


# =========================
# Training Utils
# =========================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_preview(cfg: TrainingConfig, epoch: int, batch, pred, split="train"):
    """Save preview images during training."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    outdir = os.path.join(cfg.out_dir, f"previews_{split}")
    os.makedirs(outdir, exist_ok=True)

    # pred: N x 3 x H x W in [-1,1]
    # convert to uint8 Δ (128 neutral)
    y = pred.detach().cpu()
    y01 = (y * 0.5 + 0.5).clamp(0, 1)            # [0,1]
    y255 = (y01 * 255.0).round().to(torch.uint8) # N x 3 x H x W

    tgt = batch["target"].permute(0,3,1,2).cpu()
    tgt01 = (tgt * 0.5 + 0.5).clamp(0,1)
    tgt255 = (tgt01 * 255.0).round().to(torch.uint8)

    m = batch["mask"].unsqueeze(1).cpu()

    for i in range(min(6, y.size(0))):
        pred_img = y255[i].permute(1,2,0).numpy()
        tgt_img  = tgt255[i].permute(1,2,0).numpy()
        mask_img = (m[i,0].numpy()*255).astype(np.uint8)

        Image.fromarray(pred_img).save(os.path.join(outdir, f"ep{epoch:03d}_sample{i}_pred.png"))
        Image.fromarray(tgt_img).save(os.path.join(outdir, f"ep{epoch:03d}_sample{i}_tgt.png"))
        Image.fromarray(mask_img).save(os.path.join(outdir, f"ep{epoch:03d}_sample{i}_mask.png"))


# =========================
# Training Loop
# =========================
def train_model(cfg: TrainingConfig):
    """Main training function."""
    set_seed(cfg.data_cfg.seed)
    
    print("Creating data loaders...")
    dl_tr, dl_va, in_ch = create_dataloaders(
        cfg.data_cfg, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers
    )
    
    # Model / Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = UNetSmall(in_ch=in_ch, out_ch=3, base=48).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    print(f"Model created with {in_ch} input channels")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val = float("inf")
    
    print("Starting training...")
    for ep in range(1, cfg.epochs + 1):
        # Training phase
        model.train()
        tot = 0.0
        num_batches = 0
        
        for batch in dl_tr:
            x = batch["input"].to(device)                         # (Cin) x H x W
            y = batch["target"].permute(0,3,1,2).to(device)       # N x 3 x H x W
            m = batch["mask"].unsqueeze(1).to(device)             # N x 1 x H x W

            pred = model(x)
            loss = masked_l1(pred, y, m, cfg.weight_bg, cfg.bg_neutral_weight)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot += loss.item()
            num_batches += 1

        avg_tr = tot / max(1, num_batches)

        # Validation phase
        model.eval()
        tot_v = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in dl_va:
                x = batch["input"].to(device)
                y = batch["target"].permute(0,3,1,2).to(device)
                m = batch["mask"].unsqueeze(1).to(device)
                pred = model(x)
                loss = masked_l1(pred, y, m, cfg.weight_bg, cfg.bg_neutral_weight)
                tot_v += loss.item()
                num_val_batches += 1

        avg_va = tot_v / max(1, num_val_batches)

        print(f"[Epoch {ep:03d}] train {avg_tr:.4f} | val {avg_va:.4f}")

        # Save previews
        if ep % cfg.save_every == 0:
            with torch.no_grad():
                batch = next(iter(dl_va))
                x = batch["input"].to(device)
                pred = model(x)
                save_preview(cfg, ep, batch, pred, split="val")

        # Save best model
        if avg_va < best_val:
            best_val = avg_va
            os.makedirs(cfg.out_dir, exist_ok=True)
            torch.save({
                "model": model.state_dict(), 
                "cfg": cfg.__dict__,
                "epoch": ep,
                "best_val_loss": best_val
            }, os.path.join(cfg.out_dir, "best_model.pt"))
            print(f"New best model saved! Val loss: {best_val:.4f}")

    print(f"Training complete. Best validation loss: {best_val:.4f}")
    return model, best_val


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train delta image reconstruction from CSV nodal data")
    
    # Data arguments
    parser.add_argument("--root", type=str, 
                       default=r"\DIGIT_simulation\Rendering_Network",
                       help="Project root containing data directories")
    parser.add_argument("--dz_scale", type=float, default=1e-3, 
                       help="Meters mapped to ±1 (e.g., 0.001)")
    parser.add_argument("--sigma_px", type=float, default=1.5, 
                       help="Gaussian splat sigma in pixels")
    parser.add_argument("--fourier_freqs", type=int, default=4,
                       help="Number of Fourier frequency components for positional encoding")
    parser.add_argument("--flip_y", action="store_true", 
                       help="Flip y so larger y appears nearer top of image")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_bg", type=float, default=0.1, 
                       help="Background loss weight")
    parser.add_argument("--bg_neutral_weight", type=float, default=0.02,
                       help="Background neutral loss weight")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=5, 
                       help="Save preview images every N epochs")
    parser.add_argument("--out_dir", type=str, 
                       default=r"\DIGIT_simulation\Rendering_Network\TrainingOutput",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create data config
    data_cfg = DataConfig(
        root=args.root,
        dz_scale=args.dz_scale,
        sigma_px=args.sigma_px,
        fourier_freqs=args.fourier_freqs,
        flip_y_to_image_top=args.flip_y
    )
    
    # Create training config
    train_cfg = TrainingConfig(
        data_cfg=data_cfg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_bg=args.weight_bg,
        bg_neutral_weight=args.bg_neutral_weight,
        num_workers=args.num_workers,
        save_every=args.save_every,
        out_dir=args.out_dir
    )
    
    print("Configuration:")
    print(f"  Data root: {data_cfg.root}")
    print(f"  Output dir: {train_cfg.out_dir}")
    print(f"  Epochs: {train_cfg.epochs}")
    print(f"  Batch size: {train_cfg.batch_size}")
    print(f"  Learning rate: {train_cfg.lr}")
    print(f"  DZ scale: {data_cfg.dz_scale}")
    print(f"  Sigma px: {data_cfg.sigma_px}")
    print(f"  Fourier freqs: {data_cfg.fourier_freqs}")
    print()
    
    # Train the model
    model, best_loss = train_model(train_cfg)
    
    print("Training completed successfully!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Results saved in: {train_cfg.out_dir}")
