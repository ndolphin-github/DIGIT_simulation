import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from prepare_rgb2dz_dataset import load_rgb_dz_data

# === Model Definition ===
class RGBtoDZ(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# === Training Function ===
def train_model(rgb_inputs, dz_outputs, save_path="rgb2dz_model.pth", epochs=100, batch_size=128):
    X_tensor = torch.tensor(rgb_inputs, dtype=torch.float32)
    y_tensor = torch.tensor(dz_outputs, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RGBtoDZ()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch} - Loss: {total_loss / len(loader):.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to: {save_path}")
    return model

# === Main Entry Point ===
if __name__ == "__main__":
    IMAGE_DIR = "DeltaImages_320x240"
    NODAL_DIR = "NodalDataOutput"
    INDENTER_JSON = "indenter_settings.json"
    CACHED_NPZ = "rgb_dz_dataset_20percent.npz"

    if os.path.exists(CACHED_NPZ):
        print("📥 Loading dataset from cache...")
        data = np.load(CACHED_NPZ)
        rgb_inputs = data["rgb"]
        dz_outputs = data["dz"]
    else:
        print("📦 Generating dataset (20% subset)...")
        rgb_all, dz_all = load_rgb_dz_data(IMAGE_DIR, NODAL_DIR, INDENTER_JSON)
        total_len = len(rgb_all)
        sample_len = int(0.2 * total_len)

        rgb_inputs = rgb_all[:sample_len]
        dz_outputs = dz_all[:sample_len]

        print(f"💾 Saving subset to {CACHED_NPZ}")
        np.savez(CACHED_NPZ, rgb=rgb_inputs, dz=dz_outputs)

    print("✅ Dataset size:", rgb_inputs.shape[0])
    train_model(rgb_inputs, dz_outputs)
