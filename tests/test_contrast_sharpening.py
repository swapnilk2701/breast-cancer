"""
Unit tests for Mammography Contrast Enhancement and Sharpening Module (Stage 5 Pipeline).
"""

import unittest
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless unit tests
import matplotlib.pyplot as plt

from src.contrast_sharpening import MammogramEnhancer, process_mammogram_roi


class TestMammogramEnhancer(unittest.TestCase):
    """Test suite for MammogramEnhancer class."""

    def setUp(self):
        """Set up test environment and generate synthetic mammography-like dummy images."""
        self.enhancer = MammogramEnhancer(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            blur_kernel_size=(5, 5),
            blur_sigma=1.0,
            sharpen_amount=1.2
        )
        
        # Create synthetic 227x227 8-bit image with low contrast background and a simulated lesion peak
        np.random.seed(42)
        x = np.linspace(-3, 3, 227)
        y = np.linspace(-3, 3, 227)
        xx, yy = np.meshgrid(x, y)
        gaussian_peak = np.exp(-(xx**2 + yy**2) / 2.0)
        
        # Scale background tissue to low contrast range [80, 120] + noise + central lesion peak
        base = 100 + 20 * gaussian_peak + np.random.normal(0, 3, (227, 227))
        self.dummy_image = np.clip(base, 0, 255).astype(np.uint8)

        # Add simulated microcalcifications (bright small dots)
        self.dummy_image[50, 50] = 240
        self.dummy_image[52, 51] = 235
        self.dummy_image[180, 190] = 245

    def test_output_shape_and_dtype_single_2d(self):
        """Verify output shape and dtype for 2D single image input (227, 227)."""
        output = self.enhancer.process_image(self.dummy_image)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, (227, 227))
        self.assertEqual(output.dtype, np.uint8)
        self.assertTrue(np.all(output >= 0) and np.all(output <= 255))

    def test_output_shape_single_3d(self):
        """Verify input with single channel dimension (227, 227, 1) returns (227, 227)."""
        img_3d = np.expand_dims(self.dummy_image, axis=-1)
        output = self.enhancer.process_image(img_3d)
        self.assertEqual(output.shape, (227, 227))
        self.assertEqual(output.dtype, np.uint8)

    def test_batch_processing_numpy(self):
        """Verify batch processing of (N, 227, 227) numpy array."""
        batch = np.stack([self.dummy_image] * 5, axis=0)  # Shape (5, 227, 227)
        output_batch = self.enhancer.process_batch(batch)
        self.assertEqual(output_batch.shape, (5, 227, 227))
        self.assertEqual(output_batch.dtype, np.uint8)

    def test_batch_processing_list(self):
        """Verify batch processing of a list of 2D images."""
        image_list = [self.dummy_image for _ in range(4)]
        output_batch = self.enhancer.process_batch(image_list)
        self.assertEqual(output_batch.shape, (4, 227, 227))
        self.assertEqual(output_batch.dtype, np.uint8)

    def test_clahe_contrast_enhancement(self):
        """Verify that CLAHE increases image standard deviation / local dynamic range."""
        clahe_out = self.enhancer.apply_clahe(self.dummy_image)
        self.assertEqual(clahe_out.shape, (227, 227))
        self.assertEqual(clahe_out.dtype, np.uint8)
        # CLAHE should expand low-contrast range, increasing standard deviation
        self.assertGreater(clahe_out.std(), self.dummy_image.std())

    def test_unsharp_mask_sharpness_increase(self):
        """Verify Unsharp Masking increases edge variance (Laplacian variance metric)."""
        clahe_out = self.enhancer.apply_clahe(self.dummy_image)
        sharpened_out = self.enhancer.apply_unsharp_mask(clahe_out)

        laplacian_clahe = cv2.Laplacian(clahe_out, cv2.CV_64F).var()
        laplacian_sharpened = cv2.Laplacian(sharpened_out, cv2.CV_64F).var()

        # Sharpened image must have higher high-frequency edge energy (Laplacian variance)
        self.assertGreater(laplacian_sharpened, laplacian_clahe)

    def test_invalid_dtype_error(self):
        """Verify float32 input raises TypeError."""
        float_img = self.dummy_image.astype(np.float32) / 255.0
        with self.assertRaises(TypeError):
            self.enhancer.process_image(float_img)

    def test_empty_image_error(self):
        """Verify empty array raises ValueError."""
        empty_img = np.array([], dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.enhancer.process_image(empty_img)

    def test_invalid_shape_error(self):
        """Verify 4D image passed to process_image raises ValueError."""
        invalid_img = np.zeros((227, 227, 3, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.enhancer.process_image(invalid_img)

    def test_invalid_init_parameters(self):
        """Verify invalid initialization parameters raise appropriate errors."""
        with self.assertRaises(ValueError):
            MammogramEnhancer(clip_limit=-1.0)
        with self.assertRaises(ValueError):
            MammogramEnhancer(tile_grid_size=(0, 8))
        with self.assertRaises(ValueError):
            MammogramEnhancer(blur_kernel_size=(4, 4))  # Even kernel size

    def test_visualization_function(self):
        """Verify visualization method runs without error and returns figure."""
        clahe_img, final_img, fig = self.enhancer.visualize_enhancement(self.dummy_image)
        self.assertEqual(clahe_img.shape, (227, 227))
        self.assertEqual(final_img.shape, (227, 227))
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_convenience_function(self):
        """Verify process_mammogram_roi convenience wrapper works for single & batch."""
        res_single = process_mammogram_roi(self.dummy_image)
        self.assertEqual(res_single.shape, (227, 227))

        batch = np.stack([self.dummy_image] * 3, axis=0)
        res_batch = process_mammogram_roi(batch)
        self.assertEqual(res_batch.shape, (3, 227, 227))


if __name__ == '__main__':
    unittest.main()
