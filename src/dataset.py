from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    """
    2D slice-based BraTS dataset.

    Each sample contains:
        image: 4 MRI modalities -> [4, 128, 128]
        mask:  binary tumor mask -> [128, 128]
    """

    def __init__(
        self,
        metadata_csv,
        data_root,
        split,
        image_size=(128, 128),
    ):
        self.metadata_csv = Path(metadata_csv)
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size

        # Load metadata
        self.df = pd.read_csv(self.metadata_csv)

        # Keep only the requested split
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split: {split}")

        # MRI modalities
        self.modalities = ["t1n", "t1c", "t2f", "t2w"]

    def __len__(self):
        return len(self.df)

    def _get_patient_folder(self, patient_id):
        """
        Find the raw dataset folder belonging to the patient.
        Now that patient_id is the exact folder name (e.g., BraTS-GLI-01532-000),
        we can construct the path directly.
        """
        patient_folder = self.data_root / patient_id

        if not patient_folder.exists():
            raise FileNotFoundError(
                f"No raw folder found for patient: {patient_id} at {patient_folder}"
            )

        return patient_folder

    @staticmethod
    def _normalize(image):
        """
        Normalize a single MRI slice using non-zero pixels.
        """

        mask = image > 0

        if mask.sum() == 0:
            return image.astype(np.float32)

        mean = image[mask].mean()
        std = image[mask].std()

        if std < 1e-8:
            return (image - mean).astype(np.float32)

        image = (image - mean) / std

        return image.astype(np.float32)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        patient_id = row["patient_id"]
        slice_index = int(row["slice_index"])

        # Find patient folder
        patient_folder = self._get_patient_folder(patient_id)

        # ---------------------------------------------------------
        # Load the four MRI modalities
        # ---------------------------------------------------------

        channels = []

        for modality in self.modalities:

            filename = f"{patient_folder.name}-{modality}.nii.gz"
            filepath = patient_folder / filename

            if not filepath.exists():
                raise FileNotFoundError(
                    f"Missing MRI file:\n{filepath}"
                )

            volume = nib.load(filepath).get_fdata()

            # Extract axial slice
            image_slice = volume[:, :, slice_index]

            # Normalize
            image_slice = self._normalize(image_slice)

            channels.append(image_slice)

        # [4, H, W]
        image = np.stack(channels, axis=0)

        # ---------------------------------------------------------
        # Load segmentation mask
        # ---------------------------------------------------------

        seg_filename = f"{patient_folder.name}-seg.nii.gz"
        seg_filepath = patient_folder / seg_filename

        if not seg_filepath.exists():
            raise FileNotFoundError(
                f"Missing segmentation file:\n{seg_filepath}"
            )

        segmentation = nib.load(seg_filepath).get_fdata()

        mask = segmentation[:, :, slice_index]

        # BraTS labels:
        # 0 = background
        # 1, 2, 3 = tumor regions
        #
        # Convert everything except background to tumor.
        mask = (mask > 0).astype(np.float32)

        # ---------------------------------------------------------
        # Convert to tensors
        # ---------------------------------------------------------

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        # Add batch dimension temporarily for interpolation
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Resize image
        image = F.interpolate(
            image,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )

        # Resize mask
        mask = F.interpolate(
            mask,
            size=self.image_size,
            mode="nearest",
        )

        # Remove temporary dimensions
        image = image.squeeze(0)
        mask = mask.squeeze(0)

        return image, mask