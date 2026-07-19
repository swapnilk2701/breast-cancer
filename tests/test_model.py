import unittest
import numpy as np
from src.utils import load_config
from src.model import add_noise, apply_denoising

class TestMammographyPipeline(unittest.TestCase):
    
    def setUp(self):
        # Create a mock configuration
        self.config = {
            'noise': {
                'gaussian': {'var': 0.01},
                'sp': {'amount': 0.05},
                'speckle': {'var': 0.04},
                'mixed_poisson_gaussian': {'gaussian_var': 0.01}
            },
            'denoising': {
                'median': {'kernel_size': 5},
                'gaussian': {'kernel_size': [5, 5], 'sigma': 0},
                'wiener': {'balance': 0.1},
                'bilateral': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
                'non_local_means': {'h': 10, 'template_window_size': 7, 'search_window_size': 21},
                'adaptive_median': {'s_max': 7}
            }
        }
        # Create a mock grayscale image (float64, normalized to [0, 1])
        self.test_img = np.random.rand(256, 256).astype(np.float64)

    def test_noise_addition(self):
        noise_types = ['gaussian', 's&p', 'speckle', 'poisson', 'mixed_poisson_gaussian']
        for noise in noise_types:
            noisy_img = add_noise(self.test_img, noise, self.config)
            self.assertEqual(noisy_img.shape, self.test_img.shape, f"Shape mismatch for noise: {noise}")
            self.assertTrue(np.all(noisy_img >= 0.0), f"Negative pixel values for noise: {noise}")

    def test_denoising_methods(self):
        denoising_methods = ['median', 'gaussian', 'wiener', 'bilateral', 'non_local_means', 'anscombe_wiener', 'adaptive_median']
        # Apply gaussian noise to create a noisy image
        noisy_img = add_noise(self.test_img, 'gaussian', self.config)
        
        for method in denoising_methods:
            denoised_img = apply_denoising(noisy_img, method, self.config)
            self.assertEqual(denoised_img.shape, self.test_img.shape, f"Shape mismatch for method: {method}")
            self.assertTrue(np.all(denoised_img >= 0.0), f"Negative pixel values for method: {method}")
            self.assertTrue(np.all(denoised_img <= 1.0), f"Pixel values exceeding 1.0 for method: {method}")

if __name__ == '__main__':
    unittest.main()
