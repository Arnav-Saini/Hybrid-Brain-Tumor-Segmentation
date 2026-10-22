import pandas as pd
from pathlib import Path

# Define paths
csv_path = Path(r"dataset\processed\slice_metadata.csv")
raw_dir = Path(r"dataset\raw\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")

print("Loading existing metadata...")
df = pd.read_csv(csv_path)

def resolve_full_id(base_id):
    # If the ID already has the full format (e.g., ends with -000), return it
    if len(str(base_id).split("-")) > 3:
        return base_id
        
    # Search the raw directory for matching folders
    matches = list(raw_dir.glob(f"{base_id}-*"))
    
    if len(matches) == 1:
        # 1:1 match (1,128 patients)
        return matches[0].name
    elif len(matches) > 1:
        # Multi-scan patients (00774, 00775, 00777, 01027, 01160)
        # Default to -000 to preserve consistency with existing slice indices
        for match in matches:
            if match.name.endswith("-000"):
                return match.name
        return matches[0].name
    else:
        print(f"Warning: No raw folder found for {base_id}")
        return base_id

print("Resolving full scan IDs...")
df["patient_id"] = df["patient_id"].apply(resolve_full_id)

# Save the corrected data
df.to_csv(csv_path, index=False)
print("Metadata updated successfully! Full scan IDs are now restored.")