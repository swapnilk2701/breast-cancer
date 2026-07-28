"""
Intensity Normalization Module for Mammography Preprocessing.

This module standardizes mammogram pixel intensity distributions across different
acquisition devices, X-ray detectors, and exposure conditions.

Methods Supported:
1. Min-Max Normalization ('min_max'): Linear scaling of intensities to a target range [target_min, target_max].
2. Robust Percentile Min-Max ('robust_min_max'): Clips extreme outlier intensities (e.g., 1st to 99th percentile)
   before linear scaling to prevent hot-spot artifacts.
3. Z-Score Standardization ('z_score'): Transforms pixel intensities to zero mean and unit variance.
4. Breast Tissue Z-Score ('tissue_z_score'): Computes mean and standard deviation strictly over non-zero
   foreground breast tissue pixels to prevent background 0-pixels from skewing normalization metrics.
"""

from typing import Tuple, Union, Optional, List
import numpy as np
import cv2
import matplotlib.pyplot as plt


class IntensityNormalizer:
    """
    Production-ready Intensity Normalizer for Mammography Images.
    
    Provides standardized intensity normalization methods for single images and batch arrays,
    ensuring consistent contrast and dynamic range across heterogeneous mammography datasets.
    """

    def __init__(
        self,
        method: str = "robust_min_max",
        p_low: float = 1.0,
        p_high: float = 99.0,
        target_min: float = 0.0,
        target_max: float = 255.0,
        eps: float = 1e-8
    ) -> None:
        """
        Initialize IntensityNormalizer parameters.

        Args:
            method (str): Normalization technique ('min_max', 'robust_min_max', 'z_score', 'tissue_z_score').
            p_low (float): Lower percentile for robust clipping (default: 1.0).
            p_high (float): Upper percentile for robust clipping (default: 99.0).
            target_min (float): Minimum value of normalized target range (default: 0.0).
            target_max (float): Maximum value of normalized target range (default: 255.0).
            eps (float): Epsilon value to prevent division by zero (default: 1e-8).
        """
        valid_methods = ['min_max', 'robust_min_max', 'z_score', 'tissue_z_score']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")
        if not (0.0 <= p_low < p_high <= 100.0):
            raise ValueError(f"Percentiles must satisfy 0 <= p_low < p_high <= 100, got ({p_low}, {p_high})")
        if target_min >= target_max:
            raise ValueError(f"target_min ({target_min}) must be strictly less than target_max ({target_max})")

        self.method = method
        self.p_low = p_low
        self.p_high = p_high
        self.target_min = target_min
        self.target_max = target_max
        self.eps = eps

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        """Validate input array type and dimensions."""
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Input must be a numpy.ndarray, got {type(image)}")
        if image.size == 0:
            raise ValueError("Input image array is empty.")

        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(axis=2)
        elif image.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image (H, W) or (H, W, 1), got shape {image.shape}")

        return image

    def normalize_min_max(self, image: np.ndarray) -> np.ndarray:
        """Apply linear Min-Max intensity normalization to [target_min, target_max]."""
        img_float = image.astype(np.float32)
        i_min = np.min(img_float)
        i_max = np.max(img_float)

        if abs(i_max - i_min) < self.eps:
            return np.full_like(img_float, self.target_min)

        normalized = (img_float - i_min) / (i_max - i_min + self.eps)
        scaled = normalized * (self.target_max - self.target_min) + self.target_min
        return scaled

    def normalize_robust_min_max(self, image: np.ndarray) -> np.ndarray:
        """
        Apply robust percentile Min-Max normalization.
        Clips intensity values below p_low percentile and above p_high percentile.
        """
        img_float = image.astype(np.float32)
        val_low = np.percentile(img_float, self.p_low)
        val_high = np.percentile(img_float, self.p_high)

        if abs(val_high - val_low) < self.eps:
            return np.full_like(img_float, self.target_min)

        clipped = np.clip(img_float, val_low, val_high)
        normalized = (clipped - val_low) / (val_high - val_low + self.eps)
        scaled = normalized * (self.target_max - self.target_min) + self.target_min
        return scaled

    def normalize_z_score(self, image: np.ndarray) -> np.ndarray:
        """Apply global Z-score standardization (zero mean, unit variance)."""
        img_float = image.astype(np.float32)
        mean = np.mean(img_float)
        std = np.std(img_float)

        if std < self.eps:
            return np.zeros_like(img_float)

        return (img_float - mean) / (std + self.eps)

    def normalize_tissue_z_score(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Z-score standardization strictly computed over non-zero tissue pixels.
        Prevents dark background pixels (0) from skewing mean and std values.
        """
        img_float = image.astype(np.float32)
        tissue_mask = img_float > 0

        if not np.any(tissue_mask):
            return np.zeros_like(img_float)

        tissue_pixels = img_float[tissue_mask]
        mean = np.mean(tissue_pixels)
        std = np.std(tissue_pixels)

        if std < self.eps:
            return np.zeros_like(img_float)

        normalized = np.zeros_like(img_float)
        normalized[tissue_mask] = (img_float[tissue_mask] - mean) / (std + self.eps)
        return normalized

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize a single grayscale mammogram image using the configured method.

        Args:
            image (np.ndarray): 2D image array.

        Returns:
            np.ndarray: Normalized float32 or uint8 array matching target specifications.
        """
        img_validated = self._validate_image(image)

        if self.method == 'min_max':
            norm_img = self.normalize_min_max(img_validated)
        elif self.method == 'robust_min_max':
            norm_img = self.normalize_robust_min_max(img_validated)
        elif self.method == 'z_score':
            norm_img = self.normalize_z_score(img_validated)
        elif self.method == 'tissue_z_score':
            norm_img = self.normalize_tissue_z_score(img_validated)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        # If target range is [0, 255], return uint8 format for image pipelines
        if self.target_min == 0.0 and self.target_max == 255.0 and self.method in ['min_max', 'robust_min_max']:
            return np.clip(norm_img, 0, 255).astype(np.uint8)

        return norm_img.astype(np.float32)

    def process_batch(self, images: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Process a batch of mammogram images for intensity normalization.

        Args:
            images (Union[np.ndarray, List[np.ndarray]]): Batch array (N, H, W) or list of 2D image arrays.

        Returns:
            np.ndarray: Normalized batch array of shape (N, H, W).
        """
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                results = [self.normalize_image(img) for img in images]
                return np.stack(results, axis=0)
            elif images.ndim == 4 and images.shape[3] == 1:
                results = [self.normalize_image(img.squeeze(axis=2)) for img in images]
                return np.stack(results, axis=0)
            elif images.ndim == 2:
                return self.normalize_image(images)
            else:
                raise ValueError(f"Unsupported batch array shape: {images.shape}")
        elif isinstance(images, (list, tuple)):
            if len(images) == 0:
                raise ValueError("Input image list is empty.")
            results = [self.normalize_image(img) for img in images]
            return np.stack(results, axis=0)
        else:
            raise TypeError(f"Unsupported batch type: {type(images)}")

    def visualize_normalization(
        self,
        image: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5)
    ) -> Tuple[np.ndarray, plt.Figure]:
        """
        Generate visual comparison of Original Image vs. Intensity Normalized Image + Histograms.

        Args:
            image (np.ndarray): Original mammogram (H, W).
            save_path (Optional[str]): Path to save visualization plot.
            figsize (Tuple[int, int]): Size of figure.

        Returns:
            Tuple[np.ndarray, plt.Figure]: (normalized_image, figure_handle)
        """
        img_validated = self._validate_image(image)
        norm_img = self.normalize_image(img_validated)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Panel 1: Original Image
        axes[0].imshow(img_validated, cmap='gray')
        axes[0].set_title(f"1. Original (range: [{img_validated.min():.0f}, {img_validated.max():.0f}])", fontsize=11, fontweight='bold')
        axes[0].axis('off')

        # Panel 2: Normalized Image
        axes[1].imshow(norm_img, cmap='gray')
        axes[1].set_title(f"2. Normalized [{self.method}] ({norm_img.min():.1f} to {norm_img.max():.1f})", fontsize=11, fontweight='bold')
        axes[1].axis('off')

        # Panel 3: Intensity Histograms
        axes[2].hist(img_validated.ravel(), bins=50, color='blue', alpha=0.5, label='Original')
        axes[2].hist(norm_img.ravel(), bins=50, color='red', alpha=0.5, label='Normalized')
        axes[2].set_title("3. Intensity Distribution Comparison", fontsize=11, fontweight='bold')
        axes[2].legend(loc='upper right')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        return norm_img, fig


# Functional wrapper interface
def normalize_intensity_roi(
    image: np.ndarray,
    method: str = "robust_min_max",
    p_low: float = 1.0,
    p_high: float = 99.0,
    target_min: float = 0.0,
    target_max: float = 255.0
) -> np.ndarray:
    """
    Convenience function interface to normalize single mammograms or batches.

    Args:
        image (np.ndarray): Image array (H, W) or batch (N, H, W).
        method (str): Normalization method ('min_max', 'robust_min_max', 'z_score', 'tissue_z_score').
        p_low (float): Lower percentile for robust clipping (default: 1.0).
        p_high (float): Upper percentile for robust clipping (default: 99.0).
        target_min (float): Target minimum range (default: 0.0).
        target_max (float): Target maximum range (default: 255.0).

    Returns:
        np.ndarray: Normalized image or batch array.
    """
    normalizer = IntensityNormalizer(
        method=method,
        p_low=p_low,
        p_high=p_high,
        target_min=target_min,
        target_max=target_max
    )
    if image.ndim in (3, 4) and (image.shape[0] > 1 or image.ndim == 4):
        return normalizer.process_batch(image)
    return normalizer.normalize_image(image)
