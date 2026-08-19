"""
Unit tests for Mammography Contrast Enhancement and Sharpening Module (Stage 5 Pipeline).
Verifies Section 3 techniques: Baseline HE, CLAHE, Unsharp Masking, Combined CLAHE+UM,
and evaluation metrics: PSNR, SSIM, Shannon Entropy, and Contrast Improvement Index (CII).
"""

import unittest
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless unit tests
import matplotlib.pyplot as plt

from src.contrast_sharpening.py import (
    MammogramEnhancer,
    process_mammogram_roi,
    calculate_entropy,
    calculate_image_contrast,
    calculate_contrast_improvement_index,
    evaluate_enhancement_metrics
)


class TestMammogramEnhancer(unittest.TestCase):
    """Test suite for MammogramEnhancer class and Section 3 Contrast Enhancement."""

    def setUp(self):
        """Set up test environment and generate synthetic mammography-like dummy images."""
        self.enhancer = MammogramEnhancer(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            blur_kernel_size=(5, 5),
            blur_sigma=1.0,
            sharpen_amount=1.2,
            window_size=16
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

    def test_he_baseline_enhancement(self):
        """Verify Baseline Global Histogram Equalization (HE)."""
        he_out = self.enhancer.apply_he(self.dummy_image)
        self.assertEqual(he_out.shape, (227, 227))
        self.assertEqual(he_out.dtype, np.uint8)
        # Global HE should expand histogram to span full dynamic range [0, 255]
        self.assertEqual(he_out.min(), 0)
        self.assertEqual(he_out.max(), 255)
        # Global HE has higher standard deviation than original
        self.assertGreater(he_out.std(), self.dummy_image.std())

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

    def test_combined_clahe_unsharp_mask(self):
        """Verify Combined CLAHE + UM final pipeline method."""
        combined_out = self.enhancer.apply_clahe_unsharp_mask(self.dummy_image)
        self.assertEqual(combined_out.shape, (227, 227))
        self.assertEqual(combined_out.dtype, np.uint8)
        # Combined should have both contrast expansion and high edge energy
        self.assertGreater(combined_out.std(), self.dummy_image.std())

    def test_process_image_dispatch(self):
        """Verify process_image correctly dispatches across all supported methods."""
        for m in ['he', 'clahe', 'unsharp_mask', 'clahe_unsharp_mask']:
            out = self.enhancer.process_image(self.dummy_image, method=m)
            self.assertEqual(out.shape, (227, 227))
            self.assertEqual(out.dtype, np.uint8)

        with self.assertRaises(ValueError):
            self.enhancer.process_image(self.dummy_image, method="invalid_method")

    def test_batch_processing_numpy(self):
        """Verify batch processing of (N, 227, 227) numpy array."""
        batch = np.stack([self.dummy_image] * 5, axis=0)  # Shape (5, 227, 227)
        output_batch = self.enhancer.process_batch(batch, method="clahe_unsharp_mask")
        self.assertEqual(output_batch.shape, (5, 227, 227))
        self.assertEqual(output_batch.dtype, np.uint8)

    def test_batch_processing_list(self):
        """Verify batch processing of a list of 2D images."""
        image_list = [self.dummy_image for _ in range(4)]
        output_batch = self.enhancer.process_batch(image_list, method="clahe")
        self.assertEqual(output_batch.shape, (4, 227, 227))
        self.assertEqual(output_batch.dtype, np.uint8)

    def test_entropy_metric(self):
        """Verify Shannon Entropy computation."""
        # 1. Constant image has 0 entropy
        const_img = np.full((100, 100), 128, dtype=np.uint8)
        self.assertAlmostEqual(calculate_entropy(const_img), 0.0, places=4)

        # 2. Rich image has positive entropy
        ent = calculate_entropy(self.dummy_image)
        self.assertGreater(ent, 1.0)
        self.assertLessEqual(ent, 8.0)

    def test_contrast_and_cii_metrics(self):
        """Verify Contrast and Contrast Improvement Index (CII) metrics."""
        # 1. Self-CII should be 1.0
        cii_self = calculate_contrast_improvement_index(self.dummy_image, self.dummy_image)
        self.assertAlmostEqual(cii_self, 1.0, places=2)

        # 2. CLAHE should improve contrast (CII > 1.0)
        clahe_img = self.enhancer.apply_clahe(self.dummy_image)
        cii_clahe = calculate_contrast_improvement_index(self.dummy_image, clahe_img)
        self.assertGreater(cii_clahe, 1.0)

    def test_evaluate_enhancement_metrics(self):
        """Verify evaluate_enhancement_metrics returns all required dictionary metrics."""
        combined_img = self.enhancer.apply_clahe_unsharp_mask(self.dummy_image)
        metrics = evaluate_enhancement_metrics(self.dummy_image, combined_img)

        expected_keys = [
            'PSNR', 'SSIM', 'Original_Entropy', 'Enhanced_Entropy',
            'Entropy_Delta', 'CII', 'Laplacian_Variance_Original', 'Laplacian_Variance_Enhanced'
        ]
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (float, int))

        self.assertGreater(metrics['PSNR'], 0)
        self.assertGreater(metrics['SSIM'], 0)
        self.assertGreater(metrics['CII'], 1.0)

    def test_compare_all_methods(self):
        """Verify compare_all_methods benchmarks all 5 Section 3 methods."""
        methods_dict, df_metrics = self.enhancer.compare_all_methods(self.dummy_image)

        self.assertEqual(len(methods_dict), 5)
        self.assertIn('Original', methods_dict)
        self.assertIn('HE', methods_dict)
        self.assertIn('CLAHE', methods_dict)
        self.assertIn('Unsharp_Mask', methods_dict)
        self.assertIn('CLAHE_plus_UM', methods_dict)

        self.assertIsInstance(df_metrics, pd.DataFrame)
        self.assertEqual(len(df_metrics), 5)
        self.assertIn('Method', df_metrics.columns)
        self.assertIn('PSNR', df_metrics.columns)
        self.assertIn('SSIM', df_metrics.columns)
        self.assertIn('CII', df_metrics.columns)
        self.assertIn('Enhanced_Entropy', df_metrics.columns)

    def test_parameter_sweep(self):
        """Verify parameter_sweep generates grid search evaluation metrics."""
        df_sweep = self.enhancer.parameter_sweep(
            self.dummy_image,
            clip_limits=[1.0, 2.0],
            tile_grids=[(8, 8)],
            blur_sigmas=[1.0],
            sharpen_amounts=[1.0, 1.2]
        )
        self.assertIsInstance(df_sweep, pd.DataFrame)
        self.assertEqual(len(df_sweep), 4)  # 2 * 1 * 1 * 2 = 4 combinations
        self.assertIn('Clip_Limit', df_sweep.columns)
        self.assertIn('CII', df_sweep.columns)
        self.assertIn('Enhanced_Entropy', df_sweep.columns)

    def test_visualize_enhancement_5panel(self):
        """Verify 5-panel visualization method runs without error and returns figure."""
        methods_dict, fig = self.enhancer.visualize_enhancement(self.dummy_image)
        self.assertEqual(len(methods_dict), 5)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 5)
        plt.close(fig)

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
        with self.assertRaises(ValueError):
            MammogramEnhancer(blur_sigma=-0.5)
        with self.assertRaises(ValueError):
            MammogramEnhancer(sharpen_amount=-1.0)

    def test_convenience_function(self):
        """Verify process_mammogram_roi convenience wrapper works for single & batch."""
        res_single = process_mammogram_roi(self.dummy_image, method="clahe")
        self.assertEqual(res_single.shape, (227, 227))

        batch = np.stack([self.dummy_image] * 3, axis=0)
        res_batch = process_mammogram_roi(batch, method="clahe_unsharp_mask")
        self.assertEqual(res_batch.shape, (3, 227, 227))


class TestSection3Pipeline(unittest.TestCase):
    """Test suite for Section 3 standalone module functions."""

    def setUp(self):
        np.random.seed(42)
        self.dummy_img = (np.random.rand(227, 227) * 255).astype(np.uint8)

    def test_section3_methods(self):
        from src.section3_contrast_sharpening import apply_he, apply_clahe, apply_unsharp_mask, apply_clahe_um, compute_metrics_for_method
        he_img = apply_he(self.dummy_img)
        self.assertEqual(he_img.shape, (227, 227))
        self.assertEqual(he_img.dtype, np.uint8)

        clahe_img = apply_clahe(self.dummy_img, clip_limit=3.0, tile_grid_size=(8, 8))
        self.assertEqual(clahe_img.shape, (227, 227))
        self.assertEqual(clahe_img.dtype, np.uint8)

        um_img = apply_unsharp_mask(self.dummy_img, kernel_size=(5, 5), sigma=1.0, amount=1.2)
        self.assertEqual(um_img.shape, (227, 227))
        self.assertEqual(um_img.dtype, np.uint8)

        clahe_um_img = apply_clahe_um(self.dummy_img, clip_limit=2.0, tile_grid_size=(8, 8), kernel_size=(5, 5), sigma=1.0, amount=1.2)
        self.assertEqual(clahe_um_img.shape, (227, 227))
        self.assertEqual(clahe_um_img.dtype, np.uint8)

        # Metric check
        m = compute_metrics_for_method(clahe_um_img, self.dummy_img, 'CLAHE_UM')
        self.assertIn('psnr', m)
        self.assertIn('ssim', m)
        self.assertIn('entropy', m)
        self.assertIn('cii', m)


if __name__ == '__main__':
    unittest.main()

