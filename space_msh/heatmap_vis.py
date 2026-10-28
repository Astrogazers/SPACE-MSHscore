"""Local heterogeneity map smoothing and multi-slice visual verification."""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


def create_hi_colormap() -> LinearSegmentedColormap:
    """Blue-white-red colormap centered at h_i = 0.5."""
    return LinearSegmentedColormap.from_list(
        "SPACE_hi_bwr",
        [(0.0, "#08306b"), (0.25, "#2171b5"), (0.5, "#f7f7f7"), (0.75, "#de2d26"), (1.0, "#7f0000")],
        N=256,
    )


def smooth_hi_within_neighborhood(
    hi_volume: np.ndarray,
    mask: np.ndarray,
    sigma_inplane_vox: float = 1.2,
) -> np.ndarray:
    """Normalized Gaussian smoothing inside the tumor mask for visualization."""
    values = np.asarray(hi_volume, dtype=np.float64)
    finite = (np.asarray(mask) > 0) & np.isfinite(values)
    sigma = (0.5, float(sigma_inplane_vox), float(sigma_inplane_vox))
    num = ndimage.gaussian_filter(np.where(finite, values, 0.0), sigma=sigma)
    den = ndimage.gaussian_filter(finite.astype(np.float64), sigma=sigma)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    valid = finite & (den > 1e-6)
    out[valid] = np.clip(num[valid] / den[valid], 0.0, 1.05)
    return out


def render_multislice_hi_verification_figure(
    mri_volume: np.ndarray,
    mask: np.ndarray,
    hi_volume: np.ndarray,
    morans_i: float,
    patient_id: str,
    sequence_name: str,
    output_path: Path,
    mshscore_3seq: Optional[float] = None,
    num_slices: int = 3,
    spacing_zyx: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    """Render multi-slice MRI + tumor contour and local heterogeneity h_i heatmaps."""
    image = np.asarray(mri_volume, dtype=np.float64)
    inside = np.asarray(mask) > 0
    shown_hi = smooth_hi_within_neighborhood(hi_volume, inside)

    valid_z = np.flatnonzero(np.any(inside, axis=(1, 2)))
    n_cols = min(int(num_slices), len(valid_z))
    chosen_z = valid_z[np.linspace(0, len(valid_z) - 1, n_cols, dtype=int)]

    yy, xx = np.where(np.any(inside, axis=0))
    y0, y1 = max(0, int(yy.min()) - 14), min(image.shape[1], int(yy.max()) + 15)
    x0, x1 = max(0, int(xx.min()) - 14), min(image.shape[2], int(xx.max()) + 15)
    aspect = float(spacing_zyx[1] / spacing_zyx[2])

    fig, axes = plt.subplots(2, n_cols, squeeze=False, figsize=(3.6 * n_cols + 1.0, 7.0), dpi=180)
    title = f"{patient_id} | {sequence_name} (Moran's I = {morans_i:.3f}, 1 - I = {1.0 - morans_i:.2f})"
    if mshscore_3seq is not None:
        title += f" | MSHscore = {mshscore_3seq:.3f}"
    fig.suptitle(title, fontsize=10.5, fontweight="bold", y=0.97)

    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    cmap = create_hi_colormap()
    for col, z in enumerate(chosen_z):
        plane = image[z, y0:y1, x0:x1]
        contour = inside[z, y0:y1, x0:x1]
        vmin, vmax = np.percentile(plane, [2, 98])
        for row in (0, 1):
            ax = axes[row, col]
            ax.imshow(plane, cmap="gray", vmin=vmin, vmax=vmax, aspect=aspect)
            if row == 1:
                heat = ax.imshow(shown_hi[z, y0:y1, x0:x1], cmap=cmap, norm=norm, aspect=aspect, alpha=0.88)
            if np.any(contour) and not np.all(contour):
                ax.contour(contour.astype(float), levels=[0.5], colors=["#f59e0b"], linewidths=1.2)
            ax.set_title(f"Slice {int(z)}: " + ("MRI + ROI" if row == 0 else "Local h_i"), fontsize=9.5)
            ax.axis("off")

    cax = fig.add_axes([0.22, 0.06, 0.56, 0.025])
    cb = fig.colorbar(heat, cax=cax, orientation="horizontal", extend="max")
    cb.set_ticks([0.0, 0.5, 1.0], labels=["0.0 (coherent)", "0.5", "1.0+ (heterogeneous)"])
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.04, right=0.96, hspace=0.18, wspace=0.10)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path
