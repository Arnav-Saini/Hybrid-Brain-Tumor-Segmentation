import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import BraTSDataset
from models import SimpleUNet, HybridUNet

def calculate_metrics(pred_logits, target):
    """Calculates Dice, IoU, Precision, and Recall."""
    # Convert logits to binary predictions
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    
    # Flatten tensors for calculation
    pred = pred.view(-1)
    target = target.view(-1)
    
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    
    eps = 1e-6
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    
    return dice, iou, precision, recall

def main():
    DATA_ROOT = r"dataset\raw\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    METADATA = r"dataset\processed\slice_metadata.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading models...")
    # Load CNN Baseline
    cnn_model = SimpleUNet(in_channels=4, out_channels=1).to(device)
    cnn_model.load_state_dict(torch.load("checkpoints/SimpleUNet_epoch5.pth", weights_only=True))
    cnn_model.eval()
    
    # Load Hybrid Model
    hybrid_model = HybridUNet(in_channels=4, out_channels=1).to(device)
    hybrid_model.load_state_dict(torch.load("checkpoints/HybridUNet_epoch5.pth", weights_only=True))
    hybrid_model.eval()

    # Load Dataset (Using 'train' split just for immediate visual confirmation)
    dataset = BraTSDataset(metadata_csv=METADATA, data_root=DATA_ROOT, split="train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print("Running evaluation and generating visuals...\n")
    
    # Grab one random sample
    images, masks = next(iter(dataloader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        cnn_out = cnn_model(images)
        hybrid_out = hybrid_model(images)

    # Calculate metrics
    cnn_dice, cnn_iou, cnn_prec, cnn_rec = calculate_metrics(cnn_out, masks)
    hyb_dice, hyb_iou, hyb_prec, hyb_rec = calculate_metrics(hybrid_out, masks)

    print("-" * 40)
    print("MODEL COMPARISON METRICS")
    print("-" * 40)
    print(f"Metric      | CNN Baseline | Hybrid Model")
    print(f"Dice Score  | {cnn_dice:.4f}       | {hyb_dice:.4f}")
    print(f"IoU         | {cnn_iou:.4f}       | {hyb_iou:.4f}")
    print(f"Precision   | {cnn_prec:.4f}       | {hyb_prec:.4f}")
    print(f"Recall      | {cnn_rec:.4f}       | {hyb_rec:.4f}")
    print("-" * 40)

    # Convert to numpy for visualization
    # We will visualize the T2-FLAIR channel (index 2) as the background MRI
    mri_slice = images[0, 2, :, :].cpu().numpy() 
    gt_mask = masks[0, 0, :, :].cpu().numpy()
    cnn_pred_mask = (torch.sigmoid(cnn_out[0, 0, :, :]) > 0.5).cpu().numpy()
    hyb_pred_mask = (torch.sigmoid(hybrid_out[0, 0, :, :]) > 0.5).cpu().numpy()

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(mri_slice, cmap='gray')
    axes[0].set_title("MRI (T2-FLAIR)")
    axes[0].axis('off')
    
    axes[1].imshow(mri_slice, cmap='gray')
    axes[1].imshow(gt_mask, cmap='Reds', alpha=0.5)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    axes[2].imshow(mri_slice, cmap='gray')
    axes[2].imshow(cnn_pred_mask, cmap='Blues', alpha=0.5)
    axes[2].set_title("CNN Baseline Pred")
    axes[2].axis('off')
    
    axes[3].imshow(mri_slice, cmap='gray')
    axes[3].imshow(hyb_pred_mask, cmap='Greens', alpha=0.5)
    axes[3].set_title("Hybrid Model Pred")
    axes[3].axis('off')

    plt.tight_layout()
    save_path = "comparison_plot.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved visual comparison to {save_path}")

if __name__ == "__main__":
    main()