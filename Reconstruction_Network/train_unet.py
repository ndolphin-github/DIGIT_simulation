import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from pytorch_msssim import ssim
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import csv

from unet_model import UNet
from dataset_unet import DIGITReconstructionDataset
from torch.optim.lr_scheduler import StepLR

# Path configuration
project_root = os.path.dirname(os.path.abspath(__file__))
root_data = os.path.join(project_root, "ProcessedData")
json_path = os.path.join(project_root, "indenter_settings.json")

with open(json_path, "r") as f:
    indenter_json = json.load(f)

indenter_list = [entry["name"] for entry in indenter_json]

batch_size = 4
num_epochs = 100
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

log_file = os.path.join(save_dir, "training_log.csv")
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_l1", "val_ssim", "val_total"])

# Loss function combining L1 and SSIM
def combined_loss(pred, target, alpha=0.85):
    l1 = torch.abs(pred - target).mean()
    ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
    return alpha * l1 + (1 - alpha) * ssim_loss

# Main training loop
if __name__ == "__main__":
    # === Dataset & Loader ===
    full_dataset = DIGITReconstructionDataset(root_dir=root_data, indenter_list=indenter_list)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = UNet(in_channels=3, out_channels=3).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    print(f"Training on {train_size} samples, validating on {val_size} samples.")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)
            loss = combined_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} → Avg Train Loss: {avg_loss:.6f}")

    
        model.eval()
        val_l1_total = 0
        val_ssim_total = 0

        with torch.no_grad():
            for val_input, val_target in val_loader:
                val_input = val_input.to(device)
                val_target = val_target.to(device)
                val_pred = model(val_input)

                l1 = torch.abs(val_pred - val_target).mean()
                ssim_val = ssim(val_pred, val_target, data_range=1.0, size_average=True)

                val_l1_total += l1.item()
                val_ssim_total += ssim_val.item()

        val_l1_avg = val_l1_total / len(val_loader)
        val_ssim_avg = val_ssim_total / len(val_loader)
        val_combined = 0.85 * val_l1_avg + 0.15 * (1 - val_ssim_avg)

        print(f"  Val L1: {val_l1_avg:.6f}, SSIM: {val_ssim_avg:.4f}, Combined: {val_combined:.6f}")

        # Log results
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, val_l1_avg, val_ssim_avg, val_combined])

        # Visualize and save sample predictions
        if epoch == num_epochs - 1:
            sample_input, sample_target = next(iter(val_loader))
            sample_input = sample_input.to(device)
            sample_target = sample_target.to(device)

            with torch.no_grad():
                pred = model(sample_input).cpu().squeeze(0)
            inp_np = sample_input.cpu().squeeze(0).numpy()
            pred_np = pred.numpy().transpose(1, 2, 0)
            target_np = sample_target.cpu().squeeze(0).numpy().transpose(1, 2, 0)
            dz_map = inp_np[2]

        # Save model
        torch.save(model.state_dict(), os.path.join(save_dir, f"unet_epoch{epoch+1:03d}.pt"))
        scheduler.step()
