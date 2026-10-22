import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import BraTSDataset
from models import SimpleUNet, HybridUNet

# 1. Custom Loss & Metrics
def calculate_dice(pred_logits, target, smooth=1e-6):
    """Calculates the Dice score for evaluation."""
    pred = torch.sigmoid(pred_logits)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

class DiceBCELoss(nn.Module):
    """Combines BCE (for pixel-wise classification) and Dice (for overlap/imbalance)."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1 - (2. * intersection + 1e-6) / (union + 1e-6)
        
        return bce_loss + dice_loss.mean()

# 2. Training Loop
def main():
    DATA_ROOT = r"dataset\raw\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    METADATA = r"dataset\processed\slice_metadata.csv"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load Dataset
    print("Loading training dataset...")
    train_dataset = BraTSDataset(metadata_csv=METADATA, data_root=DATA_ROOT, split="train")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

    # Initialize Model 
    model_name = "HybridUNet"
    model = HybridUNet(in_channels=4, out_channels=1).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceBCELoss()

    epochs = 5  # Start with 5 just to prove the loop works and the model learns
    os.makedirs("checkpoints", exist_ok=True)

    print(f"Starting training for {model_name}...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_dice = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_dice += calculate_dice(outputs, masks).item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        avg_train_dice = epoch_dice / len(train_loader)
        print(f"--- Epoch {epoch+1} Summary | Avg Loss: {avg_loss:.4f} | Avg Train Dice: {avg_train_dice:.4f} ---")
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/{model_name}_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}\n")

if __name__ == "__main__":
    main()