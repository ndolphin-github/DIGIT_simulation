# inference_delta_reconstruction.py
# Inference script for generating delta images from CSV nodal data using trained model
import os
import argparse
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import our data loading utilities
from data_loader import (
    DataConfig, gaussian_splat_dz_to_grid, make_coord_feats, 
    parse_step_tag, linmap
)


# =========================
# Model Architecture (Same as training)
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
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.bott  = DoubleConv(base*2, base*4)
        self.up2   = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.conv2 = DoubleConv(base*4, base*2)
        self.up1   = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.conv1 = DoubleConv(base*2, base)
        self.head  = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        b  = self.bott(p2)
        u2 = self.up2(b)
        if u2.size(-2) != d2.size(-2) or u2.size(-1) != d2.size(-1):
            u2 = F.interpolate(u2, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        c2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2)
        if u1.size(-2) != d1.size(-2) or u1.size(-1) != d1.size(-1):
            u1 = F.interpolate(u1, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        c1 = self.conv1(torch.cat([u1, d1], dim=1))
        y  = self.head(c1)
        return torch.tanh(y)


# =========================
# Inference Functions
# =========================
def load_trained_model(model_path: str, device: torch.device):
    """Load the trained model from checkpoint."""
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract config from checkpoint
    model_cfg = checkpoint.get('cfg', {})
    data_cfg = model_cfg.get('data_cfg', {})
    
    # Create data config from saved parameters
    cfg = DataConfig()
    if isinstance(data_cfg, dict):
        for key, value in data_cfg.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    
    # Create coordinate features
    coord_feats = make_coord_feats(
        cfg.img_h, cfg.img_w, cfg.x_range, cfg.y_range,
        cfg.fourier_freqs, cfg.flip_y_to_image_top
    )
    
    # Calculate input channels
    in_ch = 1 + coord_feats.size(0)
    
    # Create and load model
    model = UNetSmall(in_ch=in_ch, out_ch=3, base=48)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully with {in_ch} input channels")
    return model, cfg, coord_feats


def process_csv_to_input(csv_path: str, cfg: DataConfig, coord_feats: torch.Tensor, device: torch.device):
    """Process a single CSV file to model input tensor."""
    # Load CSV data
    df = pd.read_csv(csv_path)
    x = df["x"].to_numpy(np.float32)
    y = df["y"].to_numpy(np.float32)
    dz = df["dz"].to_numpy(np.float32)
    
    # Interpolate to grid
    dz_grid = gaussian_splat_dz_to_grid(
        x, y, dz, cfg.img_h, cfg.img_w, cfg.x_range, cfg.y_range,
        sigma_px=cfg.sigma_px, flip_y=cfg.flip_y_to_image_top
    )
    
    # Normalize
    dz_norm = np.clip(dz_grid / cfg.dz_scale, -1.0, 1.0)
    dz_t = torch.from_numpy(dz_norm).unsqueeze(0).float()  # 1xHxW
    
    # Combine with positional encoding
    inp = torch.cat([dz_t, coord_feats], dim=0).unsqueeze(0)  # 1x(Cin)xHxW
    
    return inp.to(device)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert model output tensor to image array."""
    # tensor: 1x3xHxW in [-1,1]
    img = tensor.squeeze(0).detach().cpu()  # 3xHxW
    img = (img * 0.5 + 0.5).clamp(0, 1)    # [0,1]
    img = (img * 255.0).round().to(torch.uint8)  # [0,255]
    img = img.permute(1, 2, 0).numpy()     # HxWx3
    return img


def load_background_image(bg_path: str, target_size: tuple = (100, 75)) -> Optional[np.ndarray]:
    """Load and resize background image."""
    try:
        if os.path.exists(bg_path):
            bg_img = Image.open(bg_path).convert('RGB')
            bg_img = bg_img.resize(target_size, Image.LANCZOS)
            return np.array(bg_img)
        else:
            print(f"Warning: Background image not found at {bg_path}")
            return None
    except Exception as e:
        print(f"Error loading background image: {str(e)}")
        return None


def composite_with_background(delta_img: np.ndarray, bg_img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Composite delta image with background image."""
    # Convert delta image from [0,255] to float [0,1]
    delta_float = delta_img.astype(np.float32) / 255.0
    bg_float = bg_img.astype(np.float32) / 255.0
    
    # Simple additive blending - add delta changes to background
    # Delta image centered at 128 (neutral), so subtract 0.5 to get actual delta
    delta_changes = delta_float - 0.5  # Now in [-0.5, 0.5]
    
    # Apply delta changes to background
    composite = bg_float + alpha * delta_changes
    
    # Clamp to valid range and convert back to uint8
    composite = np.clip(composite, 0, 1)
    return (composite * 255.0).astype(np.uint8)


def infer_single_csv(model, csv_path: str, cfg: DataConfig, coord_feats: torch.Tensor, 
                     device: torch.device, output_path: str, bg_img: Optional[np.ndarray] = None):
    """Run inference on a single CSV file and save the result."""
    try:
        # Process input
        input_tensor = process_csv_to_input(csv_path, cfg, coord_feats, device)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Convert to image
        img_array = tensor_to_image(output)
        
        # Check if this is step 098 and background is available
        csv_filename = os.path.basename(csv_path)
        is_step_098 = "step_098" in csv_filename or "step098" in csv_filename
        
        if is_step_098 and bg_img is not None:
            # Composite with background for step 098
            img_array = composite_with_background(img_array, bg_img, alpha=0.7)
            print(f"✓ Generated with background: {output_path}")
        else:
            print(f"✓ Generated: {output_path}")
        
        # Save image
        img = Image.fromarray(img_array)
        img.save(output_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing {csv_path}: {str(e)}")
        return False


def infer_unseen_data(model_path: str, unseen_root: str, output_root: str, device: torch.device, bg_path: Optional[str] = None):
    """Run inference on all unseen data."""
    # Load model
    model, cfg, coord_feats = load_trained_model(model_path, device)
    
    # Load background image if provided
    bg_img = None
    if bg_path:
        bg_img = load_background_image(bg_path, (cfg.img_w, cfg.img_h))
        if bg_img is not None:
            print(f"Background image loaded from: {bg_path}")
        else:
            print("Proceeding without background image")
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    # Find all CSV files in unseen data
    csv_pattern = os.path.join(unseen_root, "**", "*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    csv_files = sorted(csv_files)
    
    print(f"Found {len(csv_files)} CSV files for inference")
    
    success_count = 0
    total_count = len(csv_files)
    step_098_count = 0
    
    for csv_path in csv_files:
        # Extract relative path structure
        rel_path = os.path.relpath(csv_path, unseen_root)
        
        # Create output path with .jpg extension
        output_rel_path = Path(rel_path).with_suffix('.jpg')
        output_path = os.path.join(output_root, str(output_rel_path))
        
        # Create output subdirectory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Count step 098 files
        csv_filename = os.path.basename(csv_path)
        if "step_098" in csv_filename or "step098" in csv_filename:
            step_098_count += 1
        
        # Run inference
        if infer_single_csv(model, csv_path, cfg, coord_feats, device, output_path, bg_img):
            success_count += 1
    
    print(f"\nInference completed: {success_count}/{total_count} files processed successfully")
    if bg_img is not None:
        print(f"Step 098 files with background applied: {step_098_count}")
    print(f"Results saved in: {output_root}")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate delta images from unseen CSV data")
    
    parser.add_argument("--model_path", "-m", required=True,
                       help="Path to trained model file (best_model.pt)")
    parser.add_argument("--unseen_root", "-u", 
                       default=r"\DIGIT_simulation\Rendering_Network\UnseenNodalOutput",
                       help="Root directory containing unseen CSV files")
    parser.add_argument("--output_root", "-o",
                       default=r"DIGIT_simulation\Rendering_Network\inference_results",
                       help="Output directory for generated images")
    parser.add_argument("--device", "-d", default="auto",
                       help="Device to use: 'cpu', 'cuda', or 'auto'")
    parser.add_argument("--background", "-b", 
                       default=r"DIGIT_simulation\Rendering_Network\backgroundImage.jpg",
                       help="Path to background image (bg.jpg) for step full compositing")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Model path: {args.model_path}")
    print(f"Unseen data root: {args.unseen_root}")
    print(f"Output root: {args.output_root}")
    print(f"Background image: {args.background}")
    print()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        exit(1)
    
    # Check if unseen data directory exists
    if not os.path.exists(args.unseen_root):
        print(f"Error: Unseen data directory not found at {args.unseen_root}")
        exit(1)
    
    # Run inference
    infer_unseen_data(args.model_path, args.unseen_root, args.output_root, device, args.background)
    
    print("Inference script completed!")


# Example usage:
# python inference_delta_reconstruction.py -m best_model.pt
# python inference_delta_reconstruction.py -m best_model.pt -u UnseenNodalOutput -o generated_images
# python inference_delta_reconstruction.py -m best_model.pt -b bg.jpg  # With background for step 098
