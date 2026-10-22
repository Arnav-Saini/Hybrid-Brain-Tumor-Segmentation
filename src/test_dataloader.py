import torch
from torch.utils.data import DataLoader

from dataset import BraTSDataset


DATA_ROOT = r"dataset\raw\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
METADATA = r"dataset\processed\slice_metadata.csv"


def create_dataset(split):
    return BraTSDataset(
        metadata_csv=METADATA,
        data_root=DATA_ROOT,
        split=split,
    )


# Create datasets
train_dataset = create_dataset("train")
val_dataset = create_dataset("val")
test_dataset = create_dataset("test")


# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)


print("Dataset sizes:")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


# Get one training batch
images, masks = next(iter(train_loader))

print("\nTraining batch:")
print("Images shape:", images.shape)
print("Masks shape:", masks.shape)

print("Images dtype:", images.dtype)
print("Masks dtype:", masks.dtype)

print("Image range:", images.min().item(), "to", images.max().item())
print("Mask values:", masks.unique().tolist())


# Verify validation batch
val_images, val_masks = next(iter(val_loader))

print("\nValidation batch:")
print("Images shape:", val_images.shape)
print("Masks shape:", val_masks.shape)


# Verify test batch
test_images, test_masks = next(iter(test_loader))

print("\nTest batch:")
print("Images shape:", test_images.shape)
print("Masks shape:", test_masks.shape)


# Verify CUDA availability
print("\nCUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))