import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Your exact training data from the terminal outputs
    epochs = [1, 2, 3, 4, 5]
    
    cnn_loss = [1.5577, 1.4120, 1.3678, 1.3352, 1.3077]
    cnn_dice = [0.1831, 0.5102, 0.6526, 0.6750, 0.7007]
    
    hyb_loss = [1.5174, 1.3824, 1.3375, 1.3029, 1.2738]
    hyb_dice = [0.1936, 0.5128, 0.6663, 0.7075, 0.7002]

    # Set publication style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training Loss
    axes[0].plot(epochs, cnn_loss, label='CNN Baseline', marker='o', linestyle='--', color='blue')
    axes[0].plot(epochs, hyb_loss, label='Hybrid CNN-Transformer', marker='s', linestyle='-', color='green')
    axes[0].set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Dice-BCE Loss', fontsize=12)
    axes[0].set_xticks(epochs)
    axes[0].legend(fontsize=11)

    # Plot 2: Training Dice Score
    axes[1].plot(epochs, cnn_dice, label='CNN Baseline', marker='o', linestyle='--', color='blue')
    axes[1].plot(epochs, hyb_dice, label='Hybrid CNN-Transformer', marker='s', linestyle='-', color='green')
    axes[1].set_title('Training Dice Score Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Dice Score', fontsize=12)
    axes[1].set_xticks(epochs)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    save_path = "training_curves.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved training curves to {save_path}")

if __name__ == "__main__":
    main()