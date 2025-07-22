import os
import numpy as np
import torch
import cv2
from unet_model import UNet

# === Config ===
indenter_name = "round"  

project_root = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_root, "checkpoints", "unet_epoch100.pt")
data_dir = os.path.join(project_root, "ProcessedData", indenter_name)
output_dir = os.path.join(project_root, "InferenceResult", indenter_name)
ref_image_dir = os.path.join(project_root, "TargetImages", indenter_name)
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Model ===
model = UNet(in_channels=3, out_channels=3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Predict on All Samples ===
with torch.no_grad():
    for i in range(100):  # or range(320) if larger dataset
        name = f"{i:02d}"
        input_path = os.path.join(data_dir, f"input_{name}.npy")
        ref_image_path = os.path.join(ref_image_dir, f"{indenter_name}_000.jpg")

        if not os.path.exists(input_path) or not os.path.exists(ref_image_path):
            continue

        # === Load input
        input_np = np.load(input_path).astype(np.float32)
        input_tensor = torch.tensor(input_np).permute(2, 0, 1).unsqueeze(0).to(device)

        # === Predict ΔRGB
        pred_tensor = model(input_tensor).cpu().squeeze(0)
        pred_np = pred_tensor.permute(1, 2, 0).numpy().clip(0, 1)

        # === Reconstruct full RGB: ΔRGB + reference
        ref_img = cv2.imread(ref_image_path)
        ref_img = cv2.resize(ref_img, (input_np.shape[1], input_np.shape[0]))
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        #full_rgb = (ref_img + pred_np).clip(0, 1)
        
        boost_factor = 3.0  # ← Try values between 2 and 5
        enhanced_delta = np.clip(pred_np * boost_factor, -1.0, 1.0)

        # === Reconstruct full RGB
        full_rgb = np.clip(ref_img + enhanced_delta, 0, 1)


        # === Save result
        rgb_img = (full_rgb * 255).astype(np.uint8)
        save_path = os.path.join(output_dir, f"pred_{name}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
