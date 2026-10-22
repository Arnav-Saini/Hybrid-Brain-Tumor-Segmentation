import torch
from torch.utils.data import DataLoader
from dataset import BraTSDataset
from models import SimpleUNet

# 1. Setup Data
DATA_ROOT = r"dataset\raw\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
METADATA = r"dataset\processed\slice_metadata.csv"

print("Loading dataset...")
dataset = BraTSDataset(metadata_csv=METADATA, data_root=DATA_ROOT, split="train")

# Create a DataLoader to batch the data
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

# Fetch a single batch
images, masks = next(iter(dataloader))
print(f"\nBatch Images Shape: {images.shape}")
print(f"Batch Masks Shape: {masks.shape}")

# 2. Setup Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = SimpleUNet(in_channels=4, out_channels=1).to(device)
images = images.to(device)

# 3. Forward Pass Test
print("Running forward pass...")
with torch.no_grad():
    predictions = model(images)

print(f"Predictions Shape: {predictions.shape}")
print("Test completed successfully! The baseline is ready for training.")