import os
from pathlib import Path
import argparse
import cv2
import numpy as np
import torch
# Reuse helpers from clean_inference and model from clean_train
from Train import SimpleUNet


def load_normalization_params(data_dir):
    """Load normalization parameters"""
    params_file = os.path.join(data_dir, "normalization_params.txt")
    params = {}
    with open(params_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            params[key] = float(value)
    return params['GLOBAL_DZ_MIN'], params['GLOBAL_DZ_MAX']

def preprocess_image(img_path):
    """Preprocess input image"""
    rgb_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = rgb_img.astype(np.float32) / 255.0
    rgb_img = rgb_img.transpose(2, 0, 1)  # [3, H, W]
    return torch.tensor(rgb_img, dtype=torch.float32).unsqueeze(0)

def denormalize_dz(dz_normalized, dz_min, dz_max):
    """Denormalize dz values"""
    return dz_normalized * (dz_max - dz_min) + dz_min

def visualize_prediction(pred_np, dz_min, dz_max):
    """Create visualization of x, y, dz prediction"""
    H, W = pred_np.shape[1], pred_np.shape[2]
    
    # Denormalize dz
    dz_real = denormalize_dz(pred_np[2], dz_min, dz_max)
    
    # Create visualization
    vis_img = np.zeros((H, W, 3), dtype=np.uint8)
    
    # X channel (red) - normalize to [0, 255]
    x_norm = (pred_np[0] - pred_np[0].min()) / (pred_np[0].max() - pred_np[0].min() + 1e-8)
    vis_img[..., 0] = (x_norm * 255).astype(np.uint8)
    
    # Y channel (green) - normalize to [0, 255]
    y_norm = (pred_np[1] - pred_np[1].min()) / (pred_np[1].max() - pred_np[1].min() + 1e-8)
    vis_img[..., 1] = (y_norm * 255).astype(np.uint8)
    
    # DZ channel (blue) - use global normalization
    dz_vis = (dz_real - dz_min) / (dz_max - dz_min + 1e-8)
    dz_vis = np.clip(dz_vis, 0, 1)
    vis_img[..., 2] = (dz_vis * 255).astype(np.uint8)
    
    return vis_img, dz_real

def run_inference(model_path, data_dir, input_images_root, output_root):
    """Run inference on test images"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SimpleUNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load normalization parameters
    dz_min, dz_max = load_normalization_params(data_dir)
    print(f"Using dz range: [{dz_min:.6f}, {dz_max:.6f}]")
    
    os.makedirs(output_root, exist_ok=True)
    
    # Process test images
    for sensor in os.listdir(input_images_root):
        sensor_path = os.path.join(input_images_root, sensor)
        if not os.path.isdir(sensor_path):
            continue
        
        for indenter in os.listdir(sensor_path):
            indenter_path = os.path.join(sensor_path, indenter)
            if not os.path.isdir(indenter_path):
                continue
            
            # Create output directory
            out_dir = os.path.join(output_root, sensor, indenter)
            os.makedirs(out_dir, exist_ok=True)
            
            for img_file in sorted(os.listdir(indenter_path)):
                if not img_file.endswith('.jpg'):
                    continue
                
                step_num = img_file.split('_')[-1].split('.')[0]
                img_path = os.path.join(indenter_path, img_file)
                
                # Preprocess and predict
                X_tensor = preprocess_image(img_path).to(device)
                
                with torch.no_grad():
                    pred = model(X_tensor)
                
                pred_np = pred[0].cpu().numpy()  # [3, H, W]
                
                # Print statistics
                print(f"{img_file}:")
                print(f"  X: min={pred_np[0].min():.6f}, max={pred_np[0].max():.6f}, std={pred_np[0].std():.6f}")
                print(f"  Y: min={pred_np[1].min():.6f}, max={pred_np[1].max():.6f}, std={pred_np[1].std():.6f}")
                print(f"  DZ: min={pred_np[2].min():.6f}, max={pred_np[2].max():.6f}, std={pred_np[2].std():.6f}")
                
                # Create visualization
                vis_img, dz_real = visualize_prediction(pred_np, dz_min, dz_max)
                
                # Save results
                out_path = os.path.join(out_dir, f"pred_{step_num}.jpg")
                cv2.imwrite(out_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                
                # Save raw dz as numpy array for analysis
                dz_path = os.path.join(out_dir, f"dz_{step_num}.npy")
                np.save(dz_path, dz_real)
                
                print(f"  Saved: {out_path}")
                
                # Process only first few images for testing
                if int(step_num) > 99:
                    break
                
def visualize_prediction(pred_np, dz_min, dz_max):
    """Create visualization of x, y, dz prediction - matches clean_inference.py"""
    H, W = pred_np.shape[1], pred_np.shape[2]
    
    # Denormalize dz
    dz_real = denormalize_dz(pred_np[2], dz_min, dz_max)
    
    # Create visualization using RGB channels (same as clean_inference.py)
    vis_img = np.zeros((H, W, 3), dtype=np.uint8)
    
    # X channel (red) - normalize to [0, 255]
    x_norm = (pred_np[0] - pred_np[0].min()) / (pred_np[0].max() - pred_np[0].min() + 1e-8)
    vis_img[..., 0] = (x_norm * 255).astype(np.uint8)
    
    # Y channel (green) - normalize to [0, 255]
    y_norm = (pred_np[1] - pred_np[1].min()) / (pred_np[1].max() - pred_np[1].min() + 1e-8)
    vis_img[..., 1] = (y_norm * 255).astype(np.uint8)
    
    # DZ channel (blue) - use global normalization
    dz_vis = (dz_real - dz_min) / (dz_max - dz_min + 1e-8)
    dz_vis = np.clip(dz_vis, 0, 1)
    vis_img[..., 2] = (dz_vis * 255).astype(np.uint8)
    
    return vis_img
    """Create visualization of x, y, dz prediction - matches clean_inference.py"""
    H, W = pred_np.shape[1], pred_np.shape[2]
    
    # Denormalize dz
    dz_real = denormalize_dz(pred_np[2], dz_min, dz_max)
    
    # Create visualization using RGB channels (same as clean_inference.py)
    vis_img = np.zeros((H, W, 3), dtype=np.uint8)
    
    # X channel (red) - normalize to [0, 255]
    x_norm = (pred_np[0] - pred_np[0].min()) / (pred_np[0].max() - pred_np[0].min() + 1e-8)
    vis_img[..., 0] = (x_norm * 255).astype(np.uint8)
    
    # Y channel (green) - normalize to [0, 255]
    y_norm = (pred_np[1] - pred_np[1].min()) / (pred_np[1].max() - pred_np[1].min() + 1e-8)
    vis_img[..., 1] = (y_norm * 255).astype(np.uint8)
    
    # DZ channel (blue) - use global normalization
    dz_vis = (dz_real - dz_min) / (dz_max - dz_min + 1e-8)
    dz_vis = np.clip(dz_vis, 0, 1)
    vis_img[..., 2] = (dz_vis * 255).astype(np.uint8)
    
    return vis_img

def main():
    parser = argparse.ArgumentParser(description="Test inference with images in Test_image")
    parser.add_argument("--test_image_dir", type=str, 
                       default=r"Test_image",
                       help="Path to test image directory")
    parser.add_argument("--output_dir", type=str,
                       default=r"Test_inference_results",
                       help="Path to output directory")
    parser.add_argument("--model_path", type=str,
                       default=r"perception_unet_model.pth",
                       help="Path to the trained model")
    parser.add_argument("--data_dir", type=str,
                       default=r"data_description",
                       help="Path to data directory containing normalization_params.txt")
    parser.add_argument("--cmap", type=str, default="inferno",
                       help="Colormap for visualization (e.g., RdBu_r, inferno)")
    
    args = parser.parse_args()
    
    # Load normalization parameters
    dz_min, dz_max = load_normalization_params(args.data_dir)
    
    # Load model
    model = SimpleUNet()
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all images
    test_image_dir = Path(args.test_image_dir)
    
    total_images = 0
    successful_inferences = 0
    
    print(f"Looking for images in: {test_image_dir}")
    
    # Look for common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(test_image_dir.glob(f"*{ext}"))
        image_files.extend(test_image_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No image files found in {test_image_dir}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    for image_file in image_files:
        try:
            total_images += 1
            print(f"Processing: {image_file.name}")
            
            # Preprocess image (pass path string, not loaded image)
            input_tensor = preprocess_image(str(image_file))
            
            # Run inference
            with torch.no_grad():
                pred = model(input_tensor)
                pred_np = pred.squeeze(0).cpu().numpy()  # (3, H, W)
            
            # Visualize prediction (matches clean_inference.py)
            vis_img = visualize_prediction(pred_np, dz_min, dz_max)
            
            # Save result
            output_path = output_dir / f"{image_file.stem}_result.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            
            successful_inferences += 1
            print(f"  Saved result to: {output_path}")
            
        except Exception as e:
            print(f"  Error processing {image_file.name}: {e}")
    
    print(f"\nInference complete!")
    print(f"Total images: {total_images}")
    print(f"Successful inferences: {successful_inferences}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
