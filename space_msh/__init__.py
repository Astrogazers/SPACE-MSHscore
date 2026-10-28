"""SPACE-MSHscore: 3D MRI spatial heterogeneity quantification and survival prediction."""

from .io import load_image_and_mask, save_map_nifti
from .mshscore import (
    SequenceMoranResult,
    within_tumor_zscore,
    compute_sequence_morans_i_3d,
    compute_mshscore_multisequence,
    compute_mshscore_from_morans_i,
)
from .heatmap_vis import (
    create_hi_colormap,
    smooth_hi_within_neighborhood,
    render_multislice_hi_verification_figure,
)
from .survival_models import (
    LOCKED_MODEL_SPECS,
    REPRESENTATIVE_PATIENTS_FIG6,
    load_locked_parameters,
    compute_nomogram_points,
    predict_individual_survival,
)

__version__ = "1.0.0"
