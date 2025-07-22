import os
import json
import cv2
import numpy as np
import pandas as pd

def world_to_pixel(x_m, y_m):
    i = int(round((x_m + 0.0075) / 0.015 * 240))
    j = int(round(y_m / 0.020 * 320))
    return i, j

def load_rgb_dz_data(image_dir, nodal_dir, indenter_json_path, max_step=100):
    # Load indenter names
    with open(indenter_json_path, "r") as f:
        indenter_list = [entry["name"] for entry in json.load(f)]

    rgb_inputs = []
    dz_outputs = []

    for indenter in indenter_list:
        for step in range(max_step):
            image_name = f"{indenter}_{step:03d}.jpg"
            csv_name = f"topROI_step_{step:03d}.csv"

            image_path = os.path.join(image_dir, indenter, image_name)
            csv_path = os.path.join(nodal_dir, indenter, csv_name)

            if not os.path.exists(image_path) or not os.path.exists(csv_path):
                continue

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            df = pd.read_csv(csv_path)
            x_vals, y_vals, dz_vals = df["x"].values, df["y"].values, df["dz"].values
            dz_norm = np.clip(dz_vals / 0.0005, 0, 1)

            for x, y, dz in zip(x_vals, y_vals, dz_norm):
                i, j = world_to_pixel(x, y)
                if 0 <= i < 240 and 0 <= j < 320:
                    rgb = image[i, j] / 255.0
                    rgb_inputs.append(rgb)
                    dz_outputs.append(dz)

    rgb_inputs = np.array(rgb_inputs, dtype=np.float32)
    dz_outputs = np.array(dz_outputs, dtype=np.float32).reshape(-1, 1)

    return rgb_inputs, dz_outputs
