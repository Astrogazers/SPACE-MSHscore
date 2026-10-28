"""Unit tests for 3D MSHscore mathematical identities and Fig. 6 representative patients."""

import unittest
import numpy as np
from space_msh import (
    compute_sequence_morans_i_3d,
    compute_mshscore_from_morans_i,
    predict_individual_survival,
    REPRESENTATIVE_PATIENTS_FIG6,
)


class TestSpaceMSH(unittest.TestCase):
    def test_moran_identities_and_volume_qc(self):
        rng = np.random.default_rng(42)
        vol = rng.normal(100.0, 20.0, size=(16, 32, 32))
        zz, yy, xx = np.ogrid[:16, :32, :32]
        mask = ((zz - 8) / 5.0) ** 2 + ((yy - 16) / 10.0) ** 2 + ((xx - 16) / 10.0) ** 2 <= 1.0

        res = compute_sequence_morans_i_3d(vol, mask, spacing_zyx=(3.5, 1.0, 1.0), radius_mm=6.0)
        self.assertAlmostEqual(res.sum_z_squared, float(res.num_voxels), places=9)
        self.assertAlmostEqual(res.weighted_mean_hi, 1.0 - res.morans_i, places=10)
        self.assertGreaterEqual(res.volume_cm3, 5.0)

        with self.assertRaises(ValueError):
            small_mask = np.zeros_like(mask)
            small_mask[7:9, 15:18, 15:18] = True
            compute_sequence_morans_i_3d(vol, small_mask, spacing_zyx=(3.5, 1.0, 1.0), min_volume_cm3=5.0)

    def test_representative_patients_fig6(self):
        for key, pt in REPRESENTATIVE_PATIENTS_FIG6.items():
            exp = pt["expected"]
            scores = compute_mshscore_from_morans_i(pt["moran_t2fs"], pt["moran_dwi"], pt["moran_cet1fs"])
            self.assertAlmostEqual(round(scores["MSHscore"], 3), pt["mshscore"], places=3)
            self.assertAlmostEqual(round(scores["MSHscore_T2W"], 3), pt["mshscore_t2w"], places=3)

            pred = predict_individual_survival(
                pt["t_category"], pt["nodal_status"], pt["diameter_cm"],
                pt["histology"], pt["hemoglobin_g_dl"], pt["mshscore"], model_key="SPACE"
            )
            self.assertEqual(pred["eta_rounded_3dp"], exp["eta_space"])
            self.assertEqual(pred["risk_3yr_percentage_1dp"], exp["risk_3yr_space_pct"])
            self.assertEqual(pred["risk_group_3yr"], exp["risk_group_space"])
            self.assertEqual(
                [pred["pfs_percentages_integer"][m] for m in (12, 24, 36, 60)],
                exp["pfs_1_2_3_5_pct"],
            )
            self.assertEqual(pred["nomogram_points"]["msh_points"], exp["msh_points"])
            self.assertEqual(pred["nomogram_points"]["clinical_points"], exp["clinical_points"])
            self.assertEqual(pred["nomogram_points"]["total_points"], exp["total_points"])

            pred_t2 = predict_individual_survival(
                pt["t_category"], pt["nodal_status"], pt["diameter_cm"],
                pt["histology"], pt["hemoglobin_g_dl"], pt["mshscore_t2w"], model_key="SPACE-T2"
            )
            self.assertEqual(pred_t2["eta_rounded_3dp"], exp["eta_space_t2"])
            self.assertEqual(pred_t2["risk_3yr_percentage_1dp"], exp["risk_3yr_space_t2_pct"])


if __name__ == "__main__":
    unittest.main()
