"""Unit tests for image/mask geometry and finiteness checks in load_image_and_mask."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from space_msh import load_image_and_mask

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None


def _affine(spacing_xyz=(1.0, 1.0, 3.5), origin_xyz=(-50.0, -60.0, -20.0)):
    aff = np.diag([*spacing_xyz, 1.0])
    aff[:3, 3] = origin_xyz
    return aff


@unittest.skipIf(nib is None, "nibabel is required")
class TestLoadImageAndMask(unittest.TestCase):
    shape_xyz = (32, 32, 16)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        rng = np.random.default_rng(0)
        self.image = rng.normal(100.0, 20.0, size=self.shape_xyz)
        xx, yy, zz = np.ogrid[:32, :32, :16]
        self.mask = (((xx - 16) / 10.0) ** 2 + ((yy - 16) / 10.0) ** 2 + ((zz - 8) / 5.0) ** 2 <= 1.0).astype(np.uint8)

    def tearDown(self):
        self._tmp.cleanup()

    def _save(self, name, data, affine):
        path = self.tmp / name
        nib.save(nib.Nifti1Image(np.asarray(data), affine), str(path))
        return path

    def test_valid_input_passes(self):
        img = self._save("img.nii.gz", self.image, _affine())
        msk = self._save("msk.nii.gz", self.mask, _affine())
        vol, mask, spacing = load_image_and_mask(img, msk)
        self.assertEqual(vol.shape, (16, 32, 32))
        self.assertEqual(mask.shape, (16, 32, 32))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(int(mask.sum()), int(self.mask.sum()))
        self.assertEqual(spacing, (3.5, 1.0, 1.0))

    def test_tiny_float_noise_in_affine_passes(self):
        img = self._save("img.nii.gz", self.image, _affine())
        msk = self._save("msk.nii.gz", self.mask, _affine(origin_xyz=(-50.0 + 1e-5, -60.0, -20.0)))
        load_image_and_mask(img, msk)

    def test_shape_mismatch_raises(self):
        img = self._save("img.nii.gz", self.image, _affine())
        msk = self._save("msk.nii.gz", self.mask[:, :, :15], _affine())
        with self.assertRaisesRegex(ValueError, "matching 3D shapes"):
            load_image_and_mask(img, msk)

    def test_affine_shift_100mm_raises(self):
        img = self._save("img.nii.gz", self.image, _affine())
        msk = self._save("msk.nii.gz", self.mask, _affine(origin_xyz=(50.0, -60.0, -20.0)))
        with self.assertRaisesRegex(ValueError, "affines differ"):
            load_image_and_mask(img, msk)

    def test_spacing_mismatch_raises(self):
        img = self._save("img.nii.gz", self.image, _affine())
        msk = self._save("msk.nii.gz", self.mask, _affine(spacing_xyz=(1.0, 1.0, 5.0)))
        with self.assertRaisesRegex(ValueError, "spacing differ"):
            load_image_and_mask(img, msk)

    def test_orientation_mismatch_raises(self):
        img = self._save("img.nii.gz", self.image, _affine())
        flipped = _affine()
        flipped[0, 0] = -1.0
        msk = self._save("msk.nii.gz", self.mask, flipped)
        with self.assertRaisesRegex(ValueError, "orientations differ"):
            load_image_and_mask(img, msk)

    def test_nan_in_roi_raises(self):
        image = self.image.copy()
        image[16, 16, 8] = np.nan
        img = self._save("img.nii.gz", image, _affine())
        msk = self._save("msk.nii.gz", self.mask, _affine())
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            load_image_and_mask(img, msk)

    def test_inf_in_roi_raises(self):
        image = self.image.copy()
        image[16, 16, 8] = np.inf
        img = self._save("img.nii.gz", image, _affine())
        msk = self._save("msk.nii.gz", self.mask, _affine())
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            load_image_and_mask(img, msk)

    def test_nan_outside_roi_is_allowed(self):
        image = self.image.copy()
        image[0, 0, 0] = np.nan
        img = self._save("img.nii.gz", image, _affine())
        msk = self._save("msk.nii.gz", self.mask, _affine())
        vol, mask, _ = load_image_and_mask(img, msk)
        self.assertTrue(np.all(np.isfinite(vol[mask])))

    def test_empty_mask_raises(self):
        img = self._save("img.nii.gz", self.image, _affine())
        msk = self._save("msk.nii.gz", np.zeros_like(self.mask), _affine())
        with self.assertRaisesRegex(ValueError, "empty"):
            load_image_and_mask(img, msk)


if __name__ == "__main__":
    unittest.main()
