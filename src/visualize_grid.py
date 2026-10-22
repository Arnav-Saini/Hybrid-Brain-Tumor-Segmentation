import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import BraTSDataset
from models import HybridUNet

def main():
    DATA_ROOT = r"dataset\raw\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    METADATA = r"dataset\processed\slice_metadata.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We want to visualize 4 different slices
    num_samples = 4 

    print("Loading Hybrid Model...")
    hybrid_model = HybridUNet(in_channels=4, out_channels=1).to(device)
    hybrid_model.load_state_dict(torch.load("checkpoints/HybridUNet_epoch5.pth", weights_only=True))
    hybrid_model.eval()

    print("Loading dataset...")
    # Shuffle is True so you get a different random set of patients every time you run it
    dataset = BraTSDataset(metadata_csv=METADATA, data_root=DATA_ROOT, split="train")
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)

    # Grab a batch of 4 samples
    images, masks = next(iter(dataloader))
    images, masks = images.to(device), masks.to(device)

    print("Running inference...")
    with torch.no_grad():
        hybrid_out = hybrid_model(images)

    print("Generating visualization grid...")
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    # Add column headers
    cols = ["MRI (T2-FLAIR)", "Ground Truth", "Hybrid Prediction"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=14, pad=10)

    for i in range(num_samples):
        # Extract the specific slice for this row
        mri_slice = images[i, 2, :, :].cpu().numpy()  # Channel 2 is T2-FLAIR
        gt_mask = masks[i, 0, :, :].cpu().numpy()
        pred_mask = (torch.sigmoid(hybrid_out[i, 0, :, :]) > 0.5).cpu().numpy()

        # 1. Plot MRI
        axes[i, 0].imshow(mri_slice, cmap='gray')
        axes[i, 0].axis('off')
        
        # 2. Plot Ground Truth (Red Overlay)
        axes[i, 1].imshow(mri_slice, cmap='gray')
        axes[i, 1].imshow(gt_mask, cmap='Reds', alpha=0.5)
        axes[i, 1].axis('off')
        
        # 3. Plot Hybrid Prediction (Green Overlay)
        axes[i, 2].imshow(mri_slice, cmap='gray')
        axes[i, 2].imshow(pred_mask, cmap='Greens', alpha=0.5)
        axes[i, 2].axis('off')

    plt.tight_layout()
    save_path = "hybrid_results_grid.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved multi-slice visualization to {save_path}")

if __name__ == "__main__":
    main()