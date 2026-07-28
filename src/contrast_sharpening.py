"""
Mammography Contrast Enhancement and Sharpening Module (Stage 5 Pipeline).

Implements Stage 5 (Contrast Enhancement and Sharpening) for an
AI-based breast cancer detection pipeline using mammography ROI crops (227x227 8-bit PNG).
"""

from typing import Tuple, Union, Optional, List
import cv2
import numpy as np
import matplotlib.pyplot as plt


class MammogramEnhancer:
    """
    Production-ready image processor for medical mammography ROI crops.
    
    Provides contrast enhancement via CLAHE and high-frequency edge sharpening
    via Unsharp Masking for single images or image batches.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        blur_kernel_size: Tuple[int, int] = (5, 5),
        blur_sigma: float = 1.0,
        sharpen_amount: float = 1.2
    ) -> None:
        """
        Initialize the mammography contrast and sharpening pipeline parameters.

        Args:
            clip_limit (float): Threshold for contrast limiting in CLAHE (default: 2.0).
            tile_grid_size (Tuple[int, int]): Grid dimensions for CLAHE contextual tiles (default: (8, 8)).
            blur_kernel_size (Tuple[int, int]): Gaussian kernel size for Unsharp Masking (default: (5, 5)).
            blur_sigma (float): Gaussian standard deviation for Unsharp Masking (default: 1.0).
            sharpen_amount (float): Scaling factor (alpha) for high-pass detail signal (default: 1.2).
        """
        if clip_limit <= 0:
            raise ValueError(f"clip_limit must be positive, got {clip_limit}")
        if tile_grid_size[0] <= 0 or tile_grid_size[1] <= 0:
            raise ValueError(f"tile_grid_size dimensions must be positive, got {tile_grid_size}")
        if blur_kernel_size[0] % 2 == 0 or blur_kernel_size[1] % 2 == 0 or blur_kernel_size[0] <= 0 or blur_kernel_size[1] <= 0:
            raise ValueError(f"blur_kernel_size must contain odd positive integers, got {blur_kernel_size}")
        if blur_sigma <= 0:
            raise ValueError(f"blur_sigma must be positive, got {blur_sigma}")
        if sharpen_amount < 0:
            raise ValueError(f"sharpen_amount must be non-negative, got {sharpen_amount}")

        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.sharpen_amount = sharpen_amount

        # Pre-instantiate CLAHE object for optimal execution speed
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        """Validate and normalize 2D grayscale array dimensions and data type."""
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Input must be a numpy.ndarray, got {type(image)}")
        if image.size == 0:
            raise ValueError("Input image array is empty.")
        if image.dtype != np.uint8:
            raise TypeError(f"Input image must be uint8 (8-bit grayscale), got {image.dtype}")

        # Squeeze channel dimension if (H, W, 1)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(axis=2)
        elif image.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image (H, W) or (H, W, 1), got shape {image.shape}")

        return image

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Step 1: Apply CLAHE contrast enhancement.
        Prevents noise over-amplification in background/fatty tissue.
        """
        img_validated = self._validate_image(image)
        return self._clahe.apply(img_validated)

    def apply_unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Step 2: Apply Unsharp Masking (UM) to sharpen mass margins and microcalcifications.
        """
        img_validated = self._validate_image(image)

        # 1. Compute low-pass Gaussian blur
        blurred = cv2.GaussianBlur(
            img_validated,
            ksize=self.blur_kernel_size,
            sigmaX=self.blur_sigma,
            sigmaY=self.blur_sigma
        )

        # 2. Convert to float32 to prevent underflow/overflow during high-pass subtraction
        img_float = img_validated.astype(np.float32)
        blur_float = blurred.astype(np.float32)

        # 3. High-pass detail extraction: HighPass = Original - Blur
        high_pass = img_float - blur_float

        # 4. Amplify high-pass detail: Output = Original + alpha * HighPass
        sharpened_float = img_float + self.sharpen_amount * high_pass

        # 5. Clip to 8-bit uint8 intensity range [0, 255]
        return np.clip(sharpened_float, 0, 255).astype(np.uint8)

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """Execute Stage 5 pipeline (CLAHE followed by UM) on a single image."""
        clahe_img = self.apply_clahe(image)
        return self.apply_unsharp_mask(clahe_img)

    def process_batch(self, images: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Process a batch of images shaped (N, H, W), (N, H, W, 1), or a list of 2D arrays."""
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                processed_list = [self.process_image(img) for img in images]
                return np.stack(processed_list, axis=0)
            elif images.ndim == 4 and images.shape[3] == 1:
                processed_list = [self.process_image(img.squeeze(axis=2)) for img in images]
                return np.stack(processed_list, axis=0)
            elif images.ndim == 2:
                return self.process_image(images)
            else:
                raise ValueError(f"Unsupported batch array shape: {images.shape}")
        elif isinstance(images, (list, tuple)):
            if len(images) == 0:
                raise ValueError("Input image list is empty.")
            processed_list = [self.process_image(img) for img in images]
            return np.stack(processed_list, axis=0)
        else:
            raise TypeError(f"Unsupported batch type: {type(images)}")

    def visualize_enhancement(
        self,
        original_img: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5)
    ) -> Tuple[np.ndarray, np.ndarray, plt.Figure]:
        """Generate side-by-side comparison: Original vs. CLAHE vs. CLAHE+UM."""
        img_validated = self._validate_image(original_img)
        clahe_img = self.apply_clahe(img_validated)
        final_img = self.apply_unsharp_mask(clahe_img)

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes[0].imshow(img_validated, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title("1. Original Denoised", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(clahe_img, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title(f"2. CLAHE (clip={self.clip_limit}, grid={self.tile_grid_size})", fontsize=12, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(final_img, cmap='gray', vmin=0, vmax=255)
        axes[2].set_title(f"3. CLAHE + Unsharp Mask (amount={self.sharpen_amount})", fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        return clahe_img, final_img, fig


# Functional wrapper interface
def process_mammogram_roi(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    blur_kernel_size: Tuple[int, int] = (5, 5),
    blur_sigma: float = 1.0,
    sharpen_amount: float = 1.2
) -> np.ndarray:
    """Convenience function to process single images or batches."""
    enhancer = MammogramEnhancer(
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
        sharpen_amount=sharpen_amount
    )
    if image.ndim in (3, 4) and (image.shape[0] > 1 or image.ndim == 4):
        return enhancer.process_batch(image)
    return enhancer.process_image(image)
