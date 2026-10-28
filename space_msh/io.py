"""NIfTI I/O helpers for 3D MRI volumes and tumor segmentation masks."""

from pathlib import Path
import numpy as np


def load_image_and_mask(image_path, mask_path):
    """Load 3D NIfTI image and binary tumor mask in (z, y, x) order with mm spacing."""
    import nibabel as nib

    img_nii = nib.load(str(image_path))
    msk_nii = nib.load(str(mask_path))
    if len(img_nii.shape) != 3 or img_nii.shape != msk_nii.shape:
        raise ValueError(f"Image and mask must have matching 3D shapes: {img_nii.shape} vs {msk_nii.shape}")

    spacing_xyz = tuple(float(s) for s in img_nii.header.get_zooms()[:3])
    spacing_zyx = spacing_xyz[::-1]

    volume_zyx = np.asarray(img_nii.get_fdata(dtype=np.float64)).transpose(2, 1, 0)
    mask_zyx = np.asarray(msk_nii.get_fdata(dtype=np.float64)).transpose(2, 1, 0) > 0
    if not np.any(mask_zyx):
        raise ValueError("Viable tumor mask is empty.")
    return volume_zyx, mask_zyx, spacing_zyx


def save_map_nifti(values_zyx, reference_path, output_path):
    """Save a 3D (z, y, x) map to NIfTI using the geometry of reference_path."""
    import nibabel as nib

    ref = nib.load(str(reference_path))
    arr_xyz = np.asarray(values_zyx, dtype=np.float32).transpose(2, 1, 0)
    if arr_xyz.shape != ref.shape:
        raise ValueError(f"Map shape {arr_xyz.shape} does not match reference {ref.shape}")

    header = ref.header.copy()
    header.set_data_dtype(np.float32)
    out_img = nib.Nifti1Image(arr_xyz, ref.affine, header)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))
    return out_path
