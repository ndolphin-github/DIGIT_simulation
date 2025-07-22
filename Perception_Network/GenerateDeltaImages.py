import os
import cv2
import numpy as np
import json
from tqdm import tqdm

# Directories
IMAGE_DIR = "OutputImages_320x240"
DELTA_DIR = "DeltaImages_320x240"
os.makedirs(DELTA_DIR, exist_ok=True)

# Load indenter list
with open("indenter_settings.json", "r") as f:
    INDENTER_LIST = [entry["name"] for entry in json.load(f)]

# Process each indenter
for indenter in INDENTER_LIST:
    input_path = os.path.join(IMAGE_DIR, indenter)
    output_path = os.path.join(DELTA_DIR, indenter)
    os.makedirs(output_path, exist_ok=True)

    background_img_path = os.path.join(input_path, f"{indenter}_000.jpg")
    if not os.path.exists(background_img_path):
        print(f"[WARNING] Background image missing: {background_img_path}")
        continue

    background = cv2.imread(background_img_path).astype(np.int16)

    for filename in tqdm(sorted(os.listdir(input_path)), desc=f"Processing {indenter}"):
        if not filename.endswith(".jpg") or filename == f"{indenter}_000.jpg":
            continue

        image_path = os.path.join(input_path, filename)
        img = cv2.imread(image_path).astype(np.int16)

        delta = np.clip(img - background + 128, 0, 255).astype(np.uint8)  # Shift delta to [0, 255] range
        save_path = os.path.join(output_path, filename)
        cv2.imwrite(save_path, delta)

print("All delta images saved.")
