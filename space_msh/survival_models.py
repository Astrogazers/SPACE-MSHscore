"""Locked survival models and nomogram point calculation (Supplementary Table 4)."""

from copy import deepcopy
from importlib.resources import files
import json
import math
from typing import Optional

_PARAMETERS = json.loads(
    files("space_msh").joinpath("data/model_parameters_locked.json").read_text(encoding="utf-8")
)
TRAINING_PREDICTOR_MEANS = deepcopy(_PARAMETERS["training_predictor_means"])
REPRESENTATIVE_PATIENTS_FIG6 = deepcopy(_PARAMETERS["representative_patients"])
LOCKED_MODEL_SPECS = {
    key: {
        **deepcopy(spec),
        "name": spec["description"],
        "beta": deepcopy(spec["coefficients"]),
        "S0": {int(t): value for t, value in spec["baseline_survival_S0"].items()},
    }
    for key, spec in _PARAMETERS["models"].items()
}


def load_locked_parameters() -> dict:
    """Return a copy of the locked model parameters."""
    return deepcopy(_PARAMETERS)


def compute_nomogram_points(
    t_category: str,
    nodal_status: str,
    diameter_cm: float,
    histology: str,
    hemoglobin_g_dl: float,
    mshscore: float,
) -> Optional[dict]:
    """Calculate nomogram points for the primary SPACE model (Supplementary Fig. 5)."""
    scale = _PARAMETERS["nomogram_display"]
    raw = {
        "t_category_points": scale["t_category_points"][t_category],
        "nodal_points": scale["nodal_points"][nodal_status],
        "diameter_points": (float(diameter_cm) - scale["diameter_origin_cm"]) * scale["diameter_points_per_cm"],
        "histology_points": scale["histology_points"][histology],
        "hemoglobin_points": (scale["hemoglobin_origin_g_dl"] - float(hemoglobin_g_dl)) * scale["hemoglobin_points_per_g_dl"],
    }
    lo, hi = scale["msh_axis"]
    msh_pts = round((float(mshscore) - lo) / (hi - lo) * scale["msh_max_points"], 1)
    clin_pts = round(sum(raw.values()), 1)
    return {
        **{k: round(v, 1) for k, v in raw.items()},
        "msh_points": msh_pts,
        "clinical_points": clin_pts,
        "total_points": round(msh_pts + clin_pts, 1),
    }


def predict_individual_survival(
    t_category: str,
    nodal_status: str,
    diameter_cm: float,
    histology: str,
    hemoglobin_g_dl: float,
    mshscore: Optional[float] = None,
    model_key: str = "SPACE",
) -> dict:
    """Compute linear predictor, 1/2/3/5-year PFS, and 3-year risk group."""
    if model_key not in _PARAMETERS["models"]:
        raise ValueError(f"Unknown model_key: {model_key}")
    spec = _PARAMETERS["models"][model_key]
    b = spec["coefficients"]

    z_msh = 0.0 if model_key == "Clinical" else (float(mshscore) - spec["msh_mean"]) / spec["msh_sd"]
    eta = (
        spec["centering_constant"]
        + b["T3"] * (1.0 if t_category == "T3" else 0.0)
        + b["T4"] * (1.0 if t_category == "T4" else 0.0)
        + b["Pelvic"] * (1.0 if nodal_status == "Pelvic" else 0.0)
        + b["ParaAortic"] * (1.0 if nodal_status == "Para-aortic" else 0.0)
        + b["Diameter"] * float(diameter_cm)
        + b["Nonsquamous"] * (1.0 if histology == "Nonsquamous" else 0.0)
        + b["Hemoglobin"] * float(hemoglobin_g_dl)
        + b["MSH_per_SD"] * z_msh
    )

    pfs = {int(m): float(s0) ** math.exp(eta) for m, s0 in spec["baseline_survival_S0"].items()}
    risk_3yr = 1.0 - pfs[36]
    thresholds = _PARAMETERS["risk_thresholds_3yr"]
    group = "Low" if risk_3yr < thresholds["low_upper"] else ("Intermediate" if risk_3yr < thresholds["high_lower"] else "High")

    nomogram = (
        compute_nomogram_points(t_category, nodal_status, diameter_cm, histology, hemoglobin_g_dl, mshscore)
        if model_key == "SPACE" and mshscore is not None
        else None
    )
    return {
        "model": model_key,
        "eta": eta,
        "eta_rounded_3dp": round(eta, 3),
        "pfs_probabilities": pfs,
        "pfs_percentages_integer": {m: round(100.0 * p) for m, p in pfs.items()},
        "risk_3yr": risk_3yr,
        "risk_3yr_percentage_1dp": round(100.0 * risk_3yr, 1),
        "risk_group_3yr": group,
        "nomogram_points": nomogram,
    }
