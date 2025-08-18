# data_loader.py
# Data loading utilities for delta image reconstruction from CSV nodal data
import os, re, math, glob, random
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


# =========================
# Config
# =========================
@dataclass
class DataConfig:
    root: str = r"e:\DIGIT_simulation\Rendering_Network"
    img_w: int = 100
    img_h: int = 75
    x_range: Tuple[float,float] = (-0.0075, 0.0075)   # meters
    y_range: Tuple[float,float] = (0.00, 0.02)        # meters
    flip_y_to_image_top: bool = False                 # set True if your camera y is flipped
    sigma_px: float = 1.5                              # Gaussian splat std (pixels)
    dz_scale: float = 1e-3                            # normalize dz by this value
    fourier_freqs: int = 4                            # 0 disables Fourier posenc (keep raw x,y only)
    seed: int = 42


def linmap(v, a, b, A, B):
    return (v - a) * (B - A) / (b - a + 1e-12) + A


def find_matching_image(dirpath: str, step_tag: str, object_name: str, is_mask: bool = False) -> Optional[str]:
    
    step_match = re.search(r"step_?(\d+)", step_tag)
    if not step_match:
        return None
    step_num = step_match.group(1)
    
    exts = ["png", "jpg", "jpeg"]
    for ext in exts:
        if is_mask:
            
            pattern = f"{object_name}_{step_num}_mask.{ext}"
        else:
            
            pattern = f"{object_name}_{step_num}.{ext}"
        
        full_path = os.path.join(dirpath, pattern)
        if os.path.exists(full_path):
            return full_path
    return None


def to_tensor_img_uint8(img: Image.Image, size_wh: Tuple[int,int]) -> torch.Tensor:

    w, h = size_wh
    if img.size != (w, h):
        img = img.resize((w, h), resample=Image.BILINEAR)
    arr = np.array(img)  # HxWxC or HxW
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    return torch.from_numpy(arr.astype(np.uint8))


def mask_to_tensor(mask_img: Image.Image, size_wh: Tuple[int,int]) -> torch.Tensor:

    w, h = size_wh
    if mask_img.size != (w, h):
        mask_img = mask_img.resize((w, h), resample=Image.NEAREST)
    m = np.array(mask_img)
    if m.ndim == 3:
        m = m[...,0]
    # binarize: treat >0 as 1
    m = (m > 0).astype(np.float32)
    return torch.from_numpy(m)


def make_coord_feats(H: int, W: int, x_range, y_range, fourier_freqs: int = 4, flip_y: bool = False) -> torch.Tensor:

    xs = np.linspace(x_range[0], x_range[1], W, dtype=np.float32)
    ys = np.linspace(y_range[0], y_range[1], H, dtype=np.float32)
    if flip_y:
        ys = ys[::-1].copy()
    X, Y = np.meshgrid(xs, ys)        # HxW in meters
    x_norm = (X - np.mean(x_range)) / (0.5*(x_range[1]-x_range[0])+1e-12)  # ~[-1,1]
    y_norm = (Y - np.mean(y_range)) / (0.5*(y_range[1]-y_range[0])+1e-12)

    feats = [x_norm, y_norm]
    if fourier_freqs > 0:
        for k in range(fourier_freqs):
            f = (2.0**k) * math.pi
            feats += [np.sin(f * x_norm), np.cos(f * x_norm),
                      np.sin(f * y_norm), np.cos(f * y_norm)]
    feats = np.stack(feats, axis=0)   # CxHxW
    return torch.from_numpy(feats.astype(np.float32))


def gaussian_splat_dz_to_grid(x, y, dz, H, W, x_range, y_range, sigma_px=1.5, flip_y=False) -> np.ndarray:
 
    # map node positions to pixel coords
    cols = linmap(x, x_range[0], x_range[1], 0, W-1)
    rows = linmap(y, y_range[0], y_range[1], 0, H-1)
    if flip_y:
        rows = (H - 1) - rows

    acc = np.zeros((H, W), dtype=np.float64)
    wgt = np.zeros((H, W), dtype=np.float64)

    radius = int(max(1, math.ceil(3 * sigma_px)))
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma_px**2))

    for r, c, v in zip(rows, cols, dz):
        ci = int(round(c)); ri = int(round(r))
        r0 = max(0, ri - radius); r1 = min(H - 1, ri + radius)
        c0 = max(0, ci - radius); c1 = min(W - 1, ci + radius)

        kr0 = r0 - (ri - radius); kc0 = c0 - (ci - radius)
        kr1 = kr0 + (r1 - r0);    kc1 = kc0 + (c1 - c0)
        k = kernel[kr0:kr1+1, kc0:kc1+1]

        acc[r0:r1+1, c0:c1+1] += k * v
        wgt[r0:r1+1, c0:c1+1] += k

    out = np.zeros_like(acc, dtype=np.float32)
    mask = wgt > 1e-12
    out[mask] = (acc[mask] / wgt[mask]).astype(np.float32)
    return out


def parse_step_tag(csv_name: str) -> Optional[str]:
    # expects ...step_XXX... or ...stepXXX... in filename
    m = re.search(r"(step_?\d+)", csv_name)
    return m.group(1) if m else None


# =========================
# Dataset
# =========================
class DzDeltaDataset(Dataset):
    def __init__(self, cfg: DataConfig, split: str = "train", split_ratio: float = 0.8):
        self.cfg = cfg
        self.samples = []  # list of tuples: (csv_path, delta_img_path, mask_path)

        # Find CSVs in NodalDataOutput subdirectories
        nodal_dir = os.path.join(cfg.root, "NodalDataOutput")
        if not os.path.exists(nodal_dir):
            raise ValueError(f"NodalDataOutput directory not found at {nodal_dir}")
            
        # Look for CSVs in subdirectories (D21119, D21242, D21273, etc.)
        csv_list = []
        for subdir in os.listdir(nodal_dir):
            subdir_path = os.path.join(nodal_dir, subdir)
            if os.path.isdir(subdir_path):
                csv_pattern = os.path.join(subdir_path, "**", "topROI_step_*.csv")
                csv_list.extend(glob.glob(csv_pattern, recursive=True))
        
        csv_list = sorted(csv_list)
        print(f"Found {len(csv_list)} CSV files")

        # Pair with images/masks
        for csv_path in csv_list:
            step_tag = parse_step_tag(os.path.basename(csv_path))
            if not step_tag:
                continue

            path_parts = csv_path.replace("\\", "/").split("/")
            sensor_id = None
            object_name = None
            
            # Find sensor_id (starts with D followed by digits)
            for i, part in enumerate(path_parts):
                if part.startswith("D") and len(part) > 1 and part[1:].isdigit():
                    sensor_id = part
                    # Object name should be the next part in the path
                    if i + 1 < len(path_parts):
                        object_name = path_parts[i + 1]
                    break
            
            if not sensor_id or not object_name:
                print(f"Skipping CSV: could not extract sensor_id and object_name from {csv_path}")
                continue

            # Look for corresponding images and masks
            delta_dir = os.path.join(cfg.root, "DeltaImages_100x75", sensor_id, object_name)
            mask_dir  = os.path.join(cfg.root, "ContactMasks_100x75", sensor_id, object_name)
            
            if not (os.path.isdir(delta_dir) and os.path.isdir(mask_dir)):
                print(f"Skipping {sensor_id}/{object_name}: missing delta or mask directory")
                continue

            img_path = find_matching_image(delta_dir, step_tag, object_name, is_mask=False)
            msk_path = find_matching_image(mask_dir, step_tag, object_name, is_mask=True)
            if img_path is None or msk_path is None:
                print(f"Skipping {sensor_id}/{object_name}/{step_tag}: missing image or mask file")
                continue

            self.samples.append((csv_path, img_path, msk_path))

        print(f"Total samples found: {len(self.samples)}")

        # deterministic split by hashing path
        random.Random(self.cfg.seed).shuffle(self.samples)
        n = len(self.samples)
        k = int(n * split_ratio)
        if split == "train":
            self.samples = self.samples[:k]
            print(f"Training samples: {len(self.samples)}")
        else:
            self.samples = self.samples[k:]
            print(f"Validation samples: {len(self.samples)}")

        # Precompute coord features CxHxW
        self.coord_feats = make_coord_feats(cfg.img_h, cfg.img_w, cfg.x_range, cfg.y_range,
                                            cfg.fourier_freqs, cfg.flip_y_to_image_top)  # C,H,W

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        cfg = self.cfg
        csv_path, img_path, msk_path = self.samples[idx]

        # --- Load nodal csv ---
        import pandas as pd
        df = pd.read_csv(csv_path)
        x = df["x"].to_numpy(np.float32)
        y = df["y"].to_numpy(np.float32)
        dz = df["dz"].to_numpy(np.float32)

        # --- Interpolate to grid (H,W) ---
        dz_grid = gaussian_splat_dz_to_grid(
            x, y, dz, cfg.img_h, cfg.img_w, cfg.x_range, cfg.y_range,
            sigma_px=cfg.sigma_px, flip_y=cfg.flip_y_to_image_top
        )  # HxW (meters)

        # Normalize dz to roughly [-1,1]
        dz_norm = np.clip(dz_grid / cfg.dz_scale, -1.0, 1.0)
        dz_t = torch.from_numpy(dz_norm).unsqueeze(0).float()  # 1xHxW

        # Positional encodings CxHxW
        coord = self.coord_feats  # float32

        # --- Load delta image & mask ---
        img_u8 = to_tensor_img_uint8(Image.open(img_path).convert("RGB"), (cfg.img_w, cfg.img_h))
        # Map Δ to [-1,1] with 128 as 0: y = 2*(Δ/255 - 0.5)
        img_f  = (img_u8.float() / 255.0) * 2.0 - 1.0

        msk = mask_to_tensor(Image.open(msk_path), (cfg.img_w, cfg.img_h))  # HxW float {0,1}

        # Input: concat dz + posenc
        inp = torch.cat([dz_t, coord], dim=0)  # (1 + Cpos) x H x W

        sample = {
            "input": inp,              # (Cin)xHxW
            "target": img_f,           # HxWx3 in [-1,1] (we'll permute later)
            "mask": msk,               # HxW
            "paths": (csv_path, img_path, msk_path),
        }
        return sample


def create_dataloaders(cfg: DataConfig, batch_size: int = 8, num_workers: int = 4, split_ratio: float = 0.85):
    """
    Create train and validation dataloaders.
    
    Args:
        cfg: Data configuration
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        split_ratio: Ratio for train/validation split
    
    Returns:
        tuple: (train_dataloader, val_dataloader, input_channels)
    """
    # Datasets
    dset_tr = DzDeltaDataset(cfg, split="train", split_ratio=split_ratio)
    dset_va = DzDeltaDataset(cfg, split="val", split_ratio=split_ratio)
    
    assert len(dset_tr) > 0 and len(dset_va) > 0, "No samples found. Check your folder structure and filenames."
    
    # Compute number of input channels: 1 (dz) + 2 (x,y) + 4*fourier_freqs (sin/cos for x & y)
    cpos = dset_tr.coord_feats.size(0)
    in_ch = 1 + cpos
    
    # Loaders
    from torch.utils.data import DataLoader
    dl_tr = DataLoader(dset_tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(dset_va, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True, drop_last=False)
    
    return dl_tr, dl_va, in_ch


if __name__ == "__main__":
    # Test the data loader
    cfg = DataConfig()
    
    print("Testing data loader...")
    train_loader, val_loader, input_channels = create_dataloaders(cfg, batch_size=4)
    
    print(f"Input channels: {input_channels}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test loading one batch
    sample_batch = next(iter(train_loader))
    print(f"Input shape: {sample_batch['input'].shape}")
    print(f"Target shape: {sample_batch['target'].shape}")
    print(f"Mask shape: {sample_batch['mask'].shape}")
    print("Data loader test completed successfully!")
