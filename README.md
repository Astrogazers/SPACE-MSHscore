# SPACE-MSHscore# SPACE-MSHscore

Python package and interactive dynamic nomogram for quantifying 3D multiparametric MRI spatial heterogeneity (`MSHscore`) and predicting progression-free survival in locally advanced cervical cancer.

## Installation

```bash
pip install .
```

Dependencies: `numpy`, `scipy`, `nibabel`, `matplotlib`.

## Quick Start

### 1. Load preprocessed 3D MRI and viable-tumor mask

```python
from space_msh import (
    load_image_and_mask,
    compute_mshscore_multisequence,
    render_multislice_hi_verification_figure,
    predict_individual_survival,
)

t2_vol, t2_mask, t2_spacing = load_image_and_mask("T2FS.nii.gz", "mask_T2FS.nii.gz")
dwi_vol, dwi_mask, dwi_spacing = load_image_and_mask("DWI_highb.nii.gz", "mask_DWI.nii.gz")
ce_vol, ce_mask, ce_spacing = load_image_and_mask("CE_T1FS.nii.gz", "mask_T2FS.nii.gz")
```

### 2. Compute 3D Moran's I and MSHscore (6-mm spherical neighborhood)

```python
results = compute_mshscore_multisequence(
    t2_vol, t2_mask, t2_spacing,
    dwi_vol, dwi_mask, dwi_spacing,
    ce_vol, ce_mask, ce_spacing,
    radius_mm=6.0,
    min_volume_cm3=5.0,
)
print(f"MSHscore: {results['MSHscore']:.3f}, MSHscore_T2W: {results['MSHscore_T2W']:.3f}")
```

### 3. Visualize voxelwise local heterogeneity ($h_i$) maps

```python
t2_res = results["sequences"]["T2FS"]
render_multislice_hi_verification_figure(
    mri_volume=t2_vol,
    mask=t2_mask,
    hi_volume=t2_res.hi_volume,
    morans_i=t2_res.morans_i,
    patient_id="Case_01",
    sequence_name="T2FS",
    output_path="Case_01_T2FS_hi_map.png",
    mshscore_3seq=results["MSHscore"],
    spacing_zyx=t2_spacing,
)
```

### 4. Predict individual progression-free survival

```python
pred = predict_individual_survival(
    t_category="T1b3-T2b",
    nodal_status="Negative",
    diameter_cm=5.4,
    histology="Squamous",
    hemoglobin_g_dl=12.2,
    mshscore=results["MSHscore"],
    model_key="SPACE",
)
print(pred["risk_3yr_percentage_1dp"], pred["risk_group_3yr"], pred["nomogram_points"])
```

## Standalone Web Risk Calculator

Open `SPACE_Dynamic_Nomogram.html` directly in any web browser for offline risk calculation and one-click reproduction of the three representative patients in Fig. 6.

## Verification Tests

```bash
python -m unittest discover -s tests -v
```
