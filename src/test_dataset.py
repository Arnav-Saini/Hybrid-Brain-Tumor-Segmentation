from dataset import BraTSDataset


DATA_ROOT = r"dataset\raw\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
METADATA = r"dataset\processed\slice_metadata.csv"


dataset = BraTSDataset(
    metadata_csv=METADATA,
    data_root=DATA_ROOT,
    split="train",
)


print("Number of samples:", len(dataset))

image, mask = dataset[0]

print("Image shape:", image.shape)
print("Image dtype:", image.dtype)
print("Image min:", image.min().item())
print("Image max:", image.max().item())

print("Mask shape:", mask.shape)
print("Mask dtype:", mask.dtype)
print("Mask unique values:", mask.unique().tolist())