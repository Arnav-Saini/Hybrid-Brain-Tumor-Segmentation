import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from dataset import BraTSDataset
from models import HybridUNet

def main():
    DATA_ROOT = r"dataset\raw\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    METADATA = r"dataset\processed\slice_metadata.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Hybrid Model...")
    model = HybridUNet(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(torch.load("checkpoints/HybridUNet_epoch5.pth", weights_only=True))
    model.eval()

    # Load a batch of data to compute the matrix
    dataset = BraTSDataset(metadata_csv=METADATA, data_root=DATA_ROOT, split="train")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    images, masks = next(iter(dataloader))
    images, masks = images.to(device), masks.to(device)

    print("Computing pixel-level confusion matrix...")
    with torch.no_grad():
        preds_logits = model(images)
        
    # Convert to binary
    preds = (torch.sigmoid(preds_logits) > 0.5).int().cpu().numpy().flatten()
    targets = masks.int().cpu().numpy().flatten()

    # Calculate TP, FP, TN, FN
    tp = np.sum((preds == 1) & (targets == 1))
    tn = np.sum((preds == 0) & (targets == 0))
    fp = np.sum((preds == 1) & (targets == 0))
    fn = np.sum((preds == 0) & (targets == 1))

    # Construct the matrix
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Normalize by row (Actual class) for percentage visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print("Plotting...")
    sns.set_theme(style="white")
    plt.figure(figsize=(7, 6))
    
    # Plot using seaborn
    ax = sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt=".2%", 
        cmap="Blues", 
        cbar=True,
        xticklabels=["Background", "Tumor"],
        yticklabels=["Background", "Tumor"],
        annot_kws={"size": 14}
    )
    
    plt.title("Hybrid Model Pixel-Level Confusion Matrix", fontsize=15, pad=15)
    plt.xlabel("Predicted Label", fontsize=13)
    plt.ylabel("Actual Label", fontsize=13)
    
    # Adjust layout and save
    plt.tight_layout()
    save_path = "confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {save_path}")

if __name__ == "__main__":
    main()