"""3D physical-neighborhood Moran's I and MSHscore calculation."""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SequenceMoranResult:
    """Single-sequence 3D Moran's I, voxelwise local maps, and neighborhood statistics."""
    sequence_name: str
    morans_i: float
    one_minus_i: float
    weighted_mean_hi: float
    num_voxels: int
    sum_z_squared: float
    total_weight_sum: float
    through_plane_weight_fraction: float
    mean_neighbor_count: float
    z_volume: np.ndarray
    lag_volume: np.ndarray
    hi_volume: np.ndarray
    wi_volume: np.ndarray
    volume_cm3: float


def within_tumor_zscore(volume: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Standardize intensities inside the tumor mask using population SD (ddof=0)."""
    img = np.asarray(volume, dtype=np.float64)
    roi = np.asarray(mask) > 0
    if img.ndim != 3 or img.shape != roi.shape:
        raise ValueError("volume and mask must be 3D arrays with identical shapes.")
    values = img[roi]
    if values.size < 2:
        raise ValueError("Mask must contain at least 2 voxels.")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=0))
    if sd <= 0:
        raise ValueError("ROI intensity variance is zero.")
    z_values = (values - mean) / sd
    z_volume = np.zeros_like(img, dtype=np.float64)
    z_volume[roi] = z_values
    return z_values, z_volume, mean, sd


def compute_sequence_morans_i_3d(
    volume: np.ndarray,
    mask: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    radius_mm: float = 6.0,
    sequence_name: str = "T2FS",
    min_volume_cm3: float = 5.0,
) -> SequenceMoranResult:
    """Compute 3D Moran's I and voxelwise local heterogeneity h_i within a 6-mm sphere."""
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    z_values, z_volume, _, _ = within_tumor_zscore(volume, mask)
    roi = np.asarray(mask) > 0
    n_vox = int(z_values.size)

    voxel_vol_cm3 = float(np.prod(spacing)) / 1000.0
    volume_cm3 = n_vox * voxel_vol_cm3
    if volume_cm3 + 1e-12 < min_volume_cm3:
        raise ValueError(
            f"{sequence_name} viable tumor volume ({volume_cm3:.2f} cm^3) is below minimum QC threshold ({min_volume_cm3:.1f} cm^3)."
        )

    coords_zyx = np.argwhere(roi)
    coords_mm = coords_zyx * spacing
    tree = cKDTree(coords_mm)
    pairs = tree.query_pairs(r=float(radius_mm), output_type="ndarray")
    if pairs.size == 0:
        raise ValueError("No voxel pairs found within radius_mm.")

    idx_i = np.concatenate([pairs[:, 0], pairs[:, 1]])
    idx_j = np.concatenate([pairs[:, 1], pairs[:, 0]])
    diff_mm = coords_mm[idx_i] - coords_mm[idx_j]
    dist_mm = np.linalg.norm(diff_mm, axis=1)
    weights = 1.0 / dist_mm

    wi = np.bincount(idx_i, weights=weights, minlength=n_vox).astype(np.float64)
    weighted_z_sum = np.bincount(idx_i, weights=weights * z_values[idx_j], minlength=n_vox).astype(np.float64)
    neighbor_counts = np.bincount(idx_i, minlength=n_vox)

    s0 = float(np.sum(weights))
    numerator = float(np.sum(weights * z_values[idx_i] * z_values[idx_j]))
    morans_i = numerator / s0

    lag_vals = np.full(n_vox, np.nan, dtype=np.float64)
    hi_vals = np.full(n_vox, np.nan, dtype=np.float64)
    valid = wi > 0
    lag_vals[valid] = weighted_z_sum[valid] / wi[valid]
    hi_vals[valid] = 1.0 - z_values[valid] * lag_vals[valid]

    weighted_mean_hi = float(np.sum(wi[valid] * hi_vals[valid]) / np.sum(wi[valid]))
    through_plane_mask = coords_zyx[idx_i, 0] != coords_zyx[idx_j, 0]
    tp_fraction = float(np.sum(weights[through_plane_mask]) / s0)

    lag_volume = np.full(volume.shape, np.nan, dtype=np.float64)
    hi_volume = np.full(volume.shape, np.nan, dtype=np.float64)
    wi_volume = np.zeros(volume.shape, dtype=np.float64)
    lag_volume[roi] = lag_vals
    hi_volume[roi] = hi_vals
    wi_volume[roi] = wi

    return SequenceMoranResult(
        sequence_name=sequence_name,
        morans_i=morans_i,
        one_minus_i=1.0 - morans_i,
        weighted_mean_hi=weighted_mean_hi,
        num_voxels=n_vox,
        sum_z_squared=float(np.sum(z_values ** 2)),
        total_weight_sum=s0,
        through_plane_weight_fraction=tp_fraction,
        mean_neighbor_count=float(np.mean(neighbor_counts)),
        z_volume=z_volume,
        lag_volume=lag_volume,
        hi_volume=hi_volume,
        wi_volume=wi_volume,
        volume_cm3=volume_cm3,
    )


def compute_mshscore_from_morans_i(i_t2fs: float, i_dwi: float, i_cet1fs: float) -> Dict[str, float]:
    """Compute 3-sequence MSHscore and single-sequence MSHscore_T2W from Moran's I."""
    return {
        "I_T2FS": float(i_t2fs),
        "I_DWI": float(i_dwi),
        "I_CET1FS": float(i_cet1fs),
        "MSHscore": 1.0 - (float(i_t2fs) + float(i_dwi) + float(i_cet1fs)) / 3.0,
        "MSHscore_T2W": 1.0 - float(i_t2fs),
    }


def compute_mshscore_multisequence(
    t2_vol: np.ndarray, t2_mask: np.ndarray, t2_spacing: Tuple[float, float, float],
    dwi_vol: np.ndarray, dwi_mask: np.ndarray, dwi_spacing: Tuple[float, float, float],
    ce_vol: np.ndarray, ce_mask: np.ndarray, ce_spacing: Tuple[float, float, float],
    radius_mm: float = 6.0,
    min_volume_cm3: float = 5.0,
) -> Dict[str, object]:
    """Compute 3-sequence MSHscore and per-sequence local heterogeneity results."""
    res_t2 = compute_sequence_morans_i_3d(t2_vol, t2_mask, t2_spacing, radius_mm, "T2FS", min_volume_cm3)
    res_dwi = compute_sequence_morans_i_3d(dwi_vol, dwi_mask, dwi_spacing, radius_mm, "DWI", min_volume_cm3)
    res_ce = compute_sequence_morans_i_3d(ce_vol, ce_mask, ce_spacing, radius_mm, "CE-T1FS", min_volume_cm3)
    summary = compute_mshscore_from_morans_i(res_t2.morans_i, res_dwi.morans_i, res_ce.morans_i)
    return {
        **summary,
        "sequences": {"T2FS": res_t2, "DWI": res_dwi, "CE-T1FS": res_ce},
    }
