import os
import cv2
import json
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from tqdm import tqdm

# === Settings ===
project_root = os.path.dirname(os.path.abspath(__file__))
image_root = os.path.join(project_root, "TargetImages")
csv_root = os.path.join(project_root, "FEM_NodalData")
output_root = os.path.join(project_root, "ProcessedData")
json_path = os.path.join(project_root, "indenter_settings.json")

# Target image resolution
img_w, img_h = 320, 240

# === Load indenter list from JSON ===
with open(json_path, "r") as f:
    indenter_json = json.load(f)

indenter_list = [entry["name"] for entry in indenter_json]

print(f"Found {len(indenter_list)} indenters from JSON: {indenter_list}")

# === Process each indenter ===
for indenter_name in indenter_list:
    print(f"\nProcessing: {indenter_name}")
    image_folder = os.path.join(image_root, indenter_name)
    csv_folder = os.path.join(csv_root, indenter_name)
    output_folder = os.path.join(output_root, indenter_name)
    os.makedirs(output_folder, exist_ok=True)

    # Load reference RGB image (step 00)
    ref_path = os.path.join(image_folder, f"{indenter_name}_000.jpg")

    ref_img = cv2.imread(ref_path)
    ref_img = cv2.resize(ref_img, (img_w, img_h))
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Load reference nodal CSV for x/y range
    ref_csv = os.path.join(csv_folder, "topROI_step_000.csv")

    ref_df = pd.read_csv(ref_csv)
    x0, y0 = ref_df["x"].values, ref_df["y"].values

    # Interpolation grid
    grid_y, grid_x = np.meshgrid(
        np.linspace(min(y0), max(y0), img_w),  # width = y
        np.linspace(min(x0), max(x0), img_h)   # height = x
    )

    # === Loop through 100 steps ===
    for i in tqdm(range(100), desc=f"  {indenter_name}"):
        step_csv = f"{i:03d}"
        step_img = f"{i:03d}"

        csv_path = os.path.join(csv_folder, f"topROI_step_{step_csv}.csv")
        image_path = os.path.join(image_folder, f"{indenter_name}_{step_img}.jpg")

        if not os.path.exists(csv_path) or not os.path.exists(image_path):
            continue

        # Load nodal CSV
        df = pd.read_csv(csv_path)
        x, y, dz = df["x"].values, df["y"].values, df["dz"].values
        points = np.stack([x, y], axis=-1)

        # Interpolate (x, y, dz)
        interp_x = griddata(points, x, (grid_x, grid_y), method='linear')
        interp_y = griddata(points, y, (grid_x, grid_y), method='linear')
        interp_dz = griddata(points, dz, (grid_x, grid_y), method='linear')

        interp_dz = np.nan_to_num(interp_dz)*10000.0  
        interp_x = np.nan_to_num(interp_x)*1000
        interp_y = np.nan_to_num(interp_y)*1000

        input_stack = np.stack([interp_x, interp_y, interp_dz], axis=-1).astype(np.float32)

        # Load and resize RGB image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (img_w, img_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Compute ΔRGB = img_i - img_00
        delta_rgb = img - ref_img

        # Save .npy files
        np.save(os.path.join(output_folder, f"input_{step_img}.npy"), input_stack)
        np.save(os.path.join(output_folder, f"target_{step_img}.npy"), delta_rgb)
