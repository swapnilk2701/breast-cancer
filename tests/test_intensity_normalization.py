"""
Unit tests for Mammography Intensity Normalization Module.
"""

import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.intensity_normalization import IntensityNormalizer, normalize_intensity_roi


class TestIntensityNormalizer(unittest.TestCase):
    """Test suite for IntensityNormalizer class."""

    def setUp(self):
        """Set up synthetic mammography test images."""
        np.random.seed(42)
        # Low contrast base image in range [50, 150] with synthetic hot-spot outlier at 255
        self.dummy_img = np.random.randint(50, 150, size=(227, 227), dtype=np.uint8)
        self.dummy_img[0, 0] = 255  # Outlier hot spot
        self.dummy_img[0, 1] = 0    # Outlier dark spot

    def test_min_max_normalization(self):
        """Verify min_max scales output strictly within target range [0, 255]."""
        normalizer = IntensityNormalizer(method="min_max", target_min=0.0, target_max=255.0)
        norm_out = normalizer.normalize_image(self.dummy_img)

        self.assertEqual(norm_out.shape, (227, 227))
        self.assertEqual(norm_out.dtype, np.uint8)
        self.assertEqual(norm_out.min(), 0)
        self.assertEqual(norm_out.max(), 255)

    def test_robust_min_max_normalization(self):
        """Verify robust_min_max clips percentile outliers and scales to target range."""
        normalizer = IntensityNormalizer(method="robust_min_max", p_low=5.0, p_high=95.0)
        norm_out = normalizer.normalize_image(self.dummy_img)

        self.assertEqual(norm_out.shape, (227, 227))
        self.assertEqual(norm_out.dtype, np.uint8)
        self.assertTrue(norm_out.min() >= 0)
        self.assertTrue(norm_out.max() <= 255)

    def test_z_score_normalization(self):
        """Verify z_score produces zero mean and unit variance."""
        normalizer = IntensityNormalizer(method="z_score")
        norm_out = normalizer.normalize_image(self.dummy_img)

        self.assertEqual(norm_out.shape, (227, 227))
        self.assertAlmostEqual(float(np.mean(norm_out)), 0.0, places=4)
        self.assertAlmostEqual(float(np.std(norm_out)), 1.0, places=4)

    def test_tissue_z_score_normalization(self):
        """Verify tissue_z_score calculates mean and std strictly over non-zero pixels."""
        normalizer = IntensityNormalizer(method="tissue_z_score")
        norm_out = normalizer.normalize_image(self.dummy_img)

        self.assertEqual(norm_out.shape, (227, 227))
        # Background pixel at (0, 1) was 0, should remain 0
        self.assertEqual(norm_out[0, 1], 0.0)

        # Tissue pixels (> 0) should have zero mean
        tissue_pixels = norm_out[self.dummy_img > 0]
        self.assertAlmostEqual(float(np.mean(tissue_pixels)), 0.0, places=4)

    def test_batch_processing_numpy(self):
        """Verify batch processing of numpy array (N, 227, 227)."""
        normalizer = IntensityNormalizer(method="min_max")
        batch = np.stack([self.dummy_img] * 4, axis=0)
        norm_batch = normalizer.process_batch(batch)

        self.assertEqual(norm_batch.shape, (4, 227, 227))
        self.assertEqual(norm_batch.dtype, np.uint8)

    def test_batch_processing_list(self):
        """Verify batch processing of list of 2D images."""
        normalizer = IntensityNormalizer(method="robust_min_max")
        image_list = [self.dummy_img for _ in range(3)]
        norm_batch = normalizer.process_batch(image_list)

        self.assertEqual(norm_batch.shape, (3, 227, 227))
        self.assertEqual(norm_batch.dtype, np.uint8)

    def test_invalid_parameters(self):
        """Verify invalid initialization parameters raise ValueError."""
        with self.assertRaises(ValueError):
            IntensityNormalizer(method="invalid_method")
        with self.assertRaises(ValueError):
            IntensityNormalizer(p_low=95.0, p_high=5.0)  # Invalid percentile order
        with self.assertRaises(ValueError):
            IntensityNormalizer(target_min=255.0, target_max=0.0)  # Invalid target range

    def test_visualization_function(self):
        """Verify visualization method returns normalized image and matplotlib figure."""
        normalizer = IntensityNormalizer(method="robust_min_max")
        norm_img, fig = normalizer.visualize_normalization(self.dummy_img)

        self.assertEqual(norm_img.shape, (227, 227))
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_convenience_function(self):
        """Verify normalize_intensity_roi convenience function works for single and batch."""
        res_single = normalize_intensity_roi(self.dummy_img, method="min_max")
        self.assertEqual(res_single.shape, (227, 227))

        batch = np.stack([self.dummy_img] * 3, axis=0)
        res_batch = normalize_intensity_roi(batch, method="robust_min_max")
        self.assertEqual(res_batch.shape, (3, 227, 227))


if __name__ == '__main__':
    unittest.main()
