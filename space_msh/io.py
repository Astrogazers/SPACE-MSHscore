"""NIfTI I/O helpers for 3D MRI volumes and tumor segmentation masks."""

from pathlib import Path
import numpy as np


GEOMETRY_ATOL_MM = 1e-3


def _check_matching_geometry(img_nii, msk_nii, atol_mm=GEOMETRY_ATOL_MM):
    """Raise ValueError unless image and mask share voxel spacing, orientation and affine."""
    import nibabel as nib

    img_aff = np.asarray(img_nii.affine, dtype=np.float64)
    msk_aff = np.asarray(msk_nii.affine, dtype=np.float64)

    img_zooms = np.asarray(img_nii.header.get_zooms()[:3], dtype=np.float64)
    msk_zooms = np.asarray(msk_nii.header.get_zooms()[:3], dtype=np.float64)
    if not np.allclose(img_zooms, msk_zooms, rtol=0.0, atol=atol_mm):
        raise ValueError(
            f"Image and mask voxel spacing differ: {tuple(img_zooms)} vs {tuple(msk_zooms)} mm. "
            "The mask must be defined on the same voxel grid as its image."
        )

    img_axes = nib.aff2axcodes(img_aff)
    msk_axes = nib.aff2axcodes(msk_aff)
    if img_axes != msk_axes:
        raise ValueError(
            f"Image and mask orientations differ: {''.join(img_axes)} vs {''.join(msk_axes)}. "
            "The mask must be defined on the same voxel grid as its image."
        )

    if not np.allclose(img_aff, msk_aff, rtol=0.0, atol=atol_mm):
        max_diff = float(np.max(np.abs(img_aff - msk_aff)))
        raise ValueError(
            "Image and mask affines differ (max absolute difference "
            f"{max_diff:.4g} mm; tolerance {atol_mm:g} mm), so the mask is not spatially aligned "
            "with the image (e.g. different origin, direction or spacing). "
            "The mask must be defined on the same voxel grid as its image."
        )


def load_image_and_mask(image_path, mask_path, atol_mm=GEOMETRY_ATOL_MM):
    """Load 3D NIfTI image and binary tumor mask in (z, y, x) order with mm spacing.

    The image and mask must share the same array shape, voxel spacing, orientation and
    affine (within ``atol_mm``); otherwise a ValueError is raised. Image voxels inside the
    mask must be finite (no NaN/Inf). Each sequence is validated against its own mask on
    its own native grid; no resampling between sequences is performed.
    """
    import nibabel as nib

    img_nii = nib.load(str(image_path))
    msk_nii = nib.load(str(mask_path))
    if len(img_nii.shape) != 3 or img_nii.shape != msk_nii.shape:
        raise ValueError(f"Image and mask must have matching 3D shapes: {img_nii.shape} vs {msk_nii.shape}")
    _check_matching_geometry(img_nii, msk_nii, atol_mm=atol_mm)

    spacing_xyz = tuple(float(s) for s in img_nii.header.get_zooms()[:3])
    spacing_zyx = spacing_xyz[::-1]

    volume_zyx = np.asarray(img_nii.get_fdata(dtype=np.float64)).transpose(2, 1, 0)
    mask_data = np.asarray(msk_nii.get_fdata(dtype=np.float64)).transpose(2, 1, 0)
    if not np.all(np.isfinite(mask_data)):
        raise ValueError("Mask contains NaN or Inf values.")
    mask_zyx = mask_data > 0
    if not np.any(mask_zyx):
        raise ValueError("Viable tumor mask is empty.")
    n_nonfinite = int(np.count_nonzero(~np.isfinite(volume_zyx[mask_zyx])))
    if n_nonfinite:
        raise ValueError(
            f"Image contains {n_nonfinite} NaN/Inf voxel(s) inside the tumor mask; "
            "check preprocessing (e.g. bias-field correction or normalization) before computing MSHscore."
        )
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
