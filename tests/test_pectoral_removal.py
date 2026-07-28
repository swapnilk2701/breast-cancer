"""
Unit tests for Mammography Pectoral Muscle Removal Module.
"""

import unittest
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.pectoral_removal import PectoralMuscleRemover, remove_pectoral_muscle_roi


class TestPectoralMuscleRemover(unittest.TestCase):
    """Test suite for PectoralMuscleRemover class."""

    def setUp(self):
        """Set up test environment and generate synthetic mammogram images with pectoral muscle."""
        self.remover = PectoralMuscleRemover(
            roi_fraction_h=0.5,
            roi_fraction_w=0.5,
            fill_value=0
        )

        # Create synthetic 227x227 mammogram with top-left bright triangular pectoral muscle
        self.synthetic_mlo_left = np.full((227, 227), 40, dtype=np.uint8)
        # Add simulated breast tissue region (center/right)
        cv2.ellipse(self.synthetic_mlo_left, (150, 150), (90, 110), 0, 0, 360, 120, -1)
        # Add simulated bright pectoral muscle in top-left corner
        poly_top_left = np.array([[0, 0], [90, 0], [0, 90]], dtype=np.int32)
        cv2.fillPoly(self.synthetic_mlo_left, [poly_top_left], 230)

        # Create synthetic mammogram with top-right pectoral muscle
        self.synthetic_mlo_right = np.full((227, 227), 40, dtype=np.uint8)
        cv2.ellipse(self.synthetic_mlo_right, (70, 150), (90, 110), 0, 0, 360, 120, -1)
        poly_top_right = np.array([[227, 0], [137, 0], [227, 90]], dtype=np.int32)
        cv2.fillPoly(self.synthetic_mlo_right, [poly_top_right], 230)

    def test_orientation_detection(self):
        """Verify orientation detection identifies top-left vs top-right pectoral position."""
        orient_left = self.remover.detect_orientation(self.synthetic_mlo_left)
        self.assertEqual(orient_left, 'top-left')

        orient_right = self.remover.detect_orientation(self.synthetic_mlo_right)
        self.assertEqual(orient_right, 'top-right')

    def test_pectoral_removal_top_left(self):
        """Verify pectoral muscle in top-left is zeroed out in cleaned image."""
        cleaned, mask = self.remover.remove_pectoral_muscle(self.synthetic_mlo_left)
        self.assertEqual(cleaned.shape, (227, 227))
        self.assertEqual(cleaned.dtype, np.uint8)
        self.assertEqual(mask.shape, (227, 227))

        # Top-left corner (0,0) originally 230 should now be fill_value 0
        self.assertEqual(cleaned[0, 0], 0)
        self.assertEqual(mask[0, 0], 0)

    def test_pectoral_removal_top_right(self):
        """Verify pectoral muscle in top-right is zeroed out in cleaned image."""
        cleaned, mask = self.remover.remove_pectoral_muscle(self.synthetic_mlo_right)
        self.assertEqual(cleaned.shape, (227, 227))
        self.assertEqual(cleaned[0, 226], 0)
        self.assertEqual(mask[0, 226], 0)

    def test_batch_processing_numpy(self):
        """Verify batch processing of (N, 227, 227) numpy array."""
        batch = np.stack([self.synthetic_mlo_left, self.synthetic_mlo_right], axis=0)
        cleaned_batch, mask_batch = self.remover.process_batch(batch)

        self.assertEqual(cleaned_batch.shape, (2, 227, 227))
        self.assertEqual(mask_batch.shape, (2, 227, 227))
        self.assertEqual(cleaned_batch.dtype, np.uint8)

    def test_batch_processing_list(self):
        """Verify batch processing of list of 2D images."""
        image_list = [self.synthetic_mlo_left, self.synthetic_mlo_right]
        cleaned_batch, mask_batch = self.remover.process_batch(image_list)

        self.assertEqual(cleaned_batch.shape, (2, 227, 227))
        self.assertEqual(mask_batch.shape, (2, 227, 227))

    def test_invalid_dtype_error(self):
        """Verify float32 input raises TypeError."""
        float_img = self.synthetic_mlo_left.astype(np.float32)
        with self.assertRaises(TypeError):
            self.remover.remove_pectoral_muscle(float_img)

    def test_empty_image_error(self):
        """Verify empty image array raises ValueError."""
        empty_img = np.array([], dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.remover.remove_pectoral_muscle(empty_img)

    def test_invalid_shape_error(self):
        """Verify 4D image array raises ValueError."""
        invalid_img = np.zeros((227, 227, 3, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.remover.remove_pectoral_muscle(invalid_img)

    def test_invalid_init_parameters(self):
        """Verify invalid initialization parameters raise ValueError."""
        with self.assertRaises(ValueError):
            PectoralMuscleRemover(roi_fraction_h=0.0)
        with self.assertRaises(ValueError):
            PectoralMuscleRemover(roi_fraction_w=1.5)

    def test_visualization_function(self):
        """Verify visualization method returns expected outputs without errors."""
        cleaned, mask, fig = self.remover.visualize_pectoral_removal(self.synthetic_mlo_left)
        self.assertEqual(cleaned.shape, (227, 227))
        self.assertEqual(mask.shape, (227, 227))
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_convenience_function(self):
        """Verify remove_pectoral_muscle_roi convenience wrapper works."""
        res_single = remove_pectoral_muscle_roi(self.synthetic_mlo_left)
        self.assertEqual(res_single.shape, (227, 227))

        batch = np.stack([self.synthetic_mlo_left] * 3, axis=0)
        res_batch = remove_pectoral_muscle_roi(batch)
        self.assertEqual(res_batch.shape, (3, 227, 227))


if __name__ == '__main__':
    unittest.main()
