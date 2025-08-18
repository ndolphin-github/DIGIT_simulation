import os
import cv2
import numpy as np
import pandas as pd

H, W = 240, 320  # Image size

def pixel_to_world(i, j):
    """Convert pixel coordinates to world coordinates"""
    x_m = (i / H) * 0.015 - 0.0075
    y_m = (j / W) * 0.020
    return x_m, y_m

def build_xy_grid():
    """Build x,y coordinate grids"""
    x_grid = np.zeros((H, W), dtype=np.float32)
    y_grid = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            x_m, y_m = pixel_to_world(i, j)
            x_grid[i, j] = x_m
            y_grid[i, j] = y_m
    return x_grid, y_grid

def prepare_sample(mask_path, csv_path, global_dz_min, global_dz_max):
    """Prepare one training sample"""
    # Load and normalize RGB image
    rgb_img = cv2.imread(mask_path)
    if rgb_img is None:
        return None, None
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = rgb_img.astype(np.float32) / 255.0
    rgb_img = rgb_img.transpose(2, 0, 1)  # [3, H, W]
    
    # Load CSV and create dz map
    df = pd.read_csv(csv_path)
    dz_map = np.zeros((H, W), dtype=np.float32)
    for _, row in df.iterrows():
        i = int(round((row['x'] + 0.0075) / 0.015 * H))
        j = int(round(row['y'] / 0.020 * W))
        if 0 <= i < H and 0 <= j < W:
            dz_map[i, j] = row['dz'] if row['dz'] <= -0.00003 else 0.0
    
    # Build coordinate grids
    x_grid, y_grid = build_xy_grid()
    
    # Normalize dz globally
    if global_dz_max > global_dz_min:
        dz_norm = (dz_map - global_dz_min) / (global_dz_max - global_dz_min)
    else:
        dz_norm = dz_map - global_dz_min
    
    # Stack as [3, H, W]: [x, y, dz_normalized]
    gt = np.stack([x_grid, y_grid, dz_norm], axis=0)
    
    return rgb_img, gt

def find_global_dz_range(mask_root, csv_root):
    """Find global min/max dz values across all data"""
    print("Finding global dz range...")
    dz_min, dz_max = None, None
    
    for trial in os.listdir(csv_root):
        trial_csv_dir = os.path.join(csv_root, trial)
        if not os.path.isdir(trial_csv_dir):
            continue
        for patch_type in os.listdir(trial_csv_dir):
            patch_type_csv_dir = os.path.join(trial_csv_dir, patch_type)
            if not os.path.isdir(patch_type_csv_dir):
                continue
            for step in range(100):
                csv_file = f"topROI_step_{step:03d}.csv"
                csv_path = os.path.join(patch_type_csv_dir, csv_file)
                if not os.path.exists(csv_path):
                    continue
                
                df = pd.read_csv(csv_path)
                dzs = df['dz'].values
                dzs = dzs[dzs <= -0.00003]  # Filter meaningful dz values
                if dzs.size > 0:
                    cur_min, cur_max = dzs.min(), dzs.max()
                    dz_min = cur_min if dz_min is None else min(dz_min, cur_min)
                    dz_max = cur_max if dz_max is None else max(dz_max, cur_max)
    
    print(f"Global dz range: [{dz_min:.6f}, {dz_max:.6f}]")
    return dz_min, dz_max

def create_dataset(mask_root, csv_root, out_dir):
    """Create the complete dataset"""
    os.makedirs(out_dir, exist_ok=True)
    
    # Find global dz range
    global_dz_min, global_dz_max = find_global_dz_range(mask_root, csv_root)
    
    # Save normalization parameters
    with open(os.path.join(out_dir, "normalization_params.txt"), "w") as f:
        f.write(f"GLOBAL_DZ_MIN={global_dz_min}\n")
        f.write(f"GLOBAL_DZ_MAX={global_dz_max}\n")
    
    # Process all samples
    idx = 0
    for trial in os.listdir(csv_root):
        trial_csv_dir = os.path.join(csv_root, trial)
        if not os.path.isdir(trial_csv_dir):
            continue
        for patch_type in os.listdir(trial_csv_dir):
            patch_type_csv_dir = os.path.join(trial_csv_dir, patch_type)
            if not os.path.isdir(patch_type_csv_dir):
                continue
            for step in range(100):
                csv_file = f"topROI_step_{step:03d}.csv"
                csv_path = os.path.join(patch_type_csv_dir, csv_file)
                mask_path = os.path.join(mask_root, trial, patch_type, f"{patch_type}_{step:03d}.jpg")
                
                if not (os.path.exists(mask_path) and os.path.exists(csv_path)):
                    continue
                
                rgb_img, gt = prepare_sample(mask_path, csv_path, global_dz_min, global_dz_max)
                if rgb_img is not None:
                    np.save(os.path.join(out_dir, f"X_{idx:06d}.npy"), rgb_img)
                    np.save(os.path.join(out_dir, f"Y_{idx:06d}.npy"), gt)
                    idx += 1
                    
                    if idx % 100 == 0:
                        print(f"Processed {idx} samples...")
    
    print(f"Dataset created with {idx} samples")
    return idx

if __name__ == "__main__":
    mask_root = "DeltaImages_320x240_masked"
    csv_root = "NodalDataOutput"
    out_dir = "data_description"
    
    create_dataset(mask_root, csv_root, out_dir)
