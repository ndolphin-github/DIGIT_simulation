import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.pool = nn.MaxPool2d(2)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Conv2d(32, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Decoder with skip connections
        d3 = self.up3(e3)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        out = self.final(d2)
        return out

class DepthDataset(Dataset):
    def __init__(self, data_dir, max_samples=None):
        self.X_files = sorted(glob.glob(f"{data_dir}/X_*.npy"))
        self.Y_files = sorted(glob.glob(f"{data_dir}/Y_*.npy"))
        
        if max_samples:
            self.X_files = self.X_files[:max_samples]
            self.Y_files = self.Y_files[:max_samples]
        
        assert len(self.X_files) == len(self.Y_files), "Mismatch in X and Y files"
        print(f"Dataset loaded with {len(self.X_files)} samples")
    
    def __len__(self):
        return len(self.X_files)
    
    def __getitem__(self, idx):
        X = np.load(self.X_files[idx])
        Y = np.load(self.Y_files[idx])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def train_model(data_dir, epochs=50, batch_size=4, lr=1e-3, save_path="perception_unet_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = DepthDataset(data_dir, max_samples=1000)  # Start with subset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Analyze data first
    print("Analyzing dataset...")
    sample_X, sample_Y = dataset[0]
    print(f"Input shape: {sample_X.shape}")
    print(f"Target shape: {sample_Y.shape}")
    print(f"Target dz channel stats: min={sample_Y[2].min():.6f}, max={sample_Y[2].max():.6f}, mean={sample_Y[2].mean():.6f}, std={sample_Y[2].std():.6f}")
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (X_batch, Y_batch) in enumerate(dataloader):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            
            # Only train on dz channel (channel 2)
            loss = criterion(pred[:, 2:3], Y_batch[:, 2:3])
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} completed, Average Loss: {avg_loss:.6f}")
        
        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{save_path}_epoch_{epoch+1}.pth")
    
    # Save final model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")

if __name__ == "__main__":
    train_model("prepared_data")
