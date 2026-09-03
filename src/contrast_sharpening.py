"""
Mammography Contrast Enhancement and Sharpening Module (Stage 5 Pipeline).

Implements Section 3 (Contrast Enhancement and Sharpening Plan) for an
AI-based breast cancer detection pipeline using mammography ROI crops.

Techniques Implemented:
1. Histogram Equalization (HE): Baseline method used to demonstrate global contrast adjustment
   and contrast limitation benefits.
2. Contrast Limited Adaptive Histogram Equalization (CLAHE): Primary contrast enhancement method
   preventing noise over-amplification in fatty/background tissue.
3. Unsharp Masking (UM): High-frequency spatial filtering to enhance edge sharpness and microcalcification visibility.
4. Combined CLAHE + UM: Final pipeline combining tile-based local contrast equalization with unsharp edge enhancement.

Evaluation Metrics:
- PSNR (Peak Signal-to-Noise Ratio): Pixel-level fidelity against reference image.
- SSIM (Structural Similarity Index Measure): Structural/perceptual similarity preserving diagnostic morphology.
- Shannon Entropy: Information content / detail richness (higher entropy = more visible detail).
- Contrast Improvement Index (CII): Ratio of enhanced local contrast to original local contrast.
"""

from typing import Tuple, Union, Optional, List, Dict
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# =====================================================================
# Quantitative Metrics Functions for Contrast Enhancement Evaluation
# =====================================================================

def extract_roi_masks(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Extracts lesion/dense tissue ROI (foreground) and surrounding healthy tissue ROI (background).
    Formula parameters:
        μ_f (μ_ROI): Mean intensity of lesion/foreground ROI.
        μ_b (μ_background): Mean intensity of surrounding healthy background ROI.
        σ_background: Standard deviation of intensity of healthy background ROI.
    Returns:
        Tuple[fg_mask, bg_mask, mu_f, mu_b, sigma_b]
    """
    img_float = image.astype(np.float64)
    if img_float.max() > 1.0:
        img_float = img_float / 255.0

    # Non-black breast tissue mask (ignore pure background outside breast)
    breast_mask = img_float > 0.02
    if not np.any(breast_mask):
        breast_mask = np.ones_like(img_float, dtype=bool)

    breast_pixels = img_float[breast_mask]
    thresh = float(np.percentile(breast_pixels, 75))  # Top 25% intensity inside breast tissue as candidate ROI

    fg_mask = breast_mask & (img_float >= thresh)
    bg_mask = breast_mask & (img_float < thresh)

    if not np.any(fg_mask):
        fg_mask = breast_mask
    if not np.any(bg_mask):
        bg_mask = breast_mask

    mu_f = float(np.mean(img_float[fg_mask]))
    mu_b = float(np.mean(img_float[bg_mask]))
    sigma_b = float(np.std(img_float[bg_mask]))
    if sigma_b < 1e-6:
        sigma_b = 1e-6

    return fg_mask, bg_mask, mu_f, mu_b, sigma_b


def calculate_entropy(image: np.ndarray) -> float:
    """
    Computes the Shannon Entropy (information content) of an image in bits.
    
    Formula:
        H = - sum_{i} p(i) * log2(p(i))
    where p(i) is the empirical probability of intensity level i.

    Args:
        image (np.ndarray): 2D grayscale image (uint8 [0, 255] or float [0, 1]).

    Returns:
        float: Shannon entropy value in bits (typically in [0.0, 8.0] for 8-bit images).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input must be a numpy.ndarray, got {type(image)}")
    if image.size == 0:
        raise ValueError("Input image is empty.")

    # Convert to 8-bit integer representation [0, 255]
    if image.dtype != np.uint8:
        img_uint8 = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
    else:
        img_uint8 = image

    # Compute intensity histogram probability distribution
    hist, _ = np.histogram(img_uint8.ravel(), bins=256, range=(0, 256))
    total = hist.sum()
    if total == 0:
        return 0.0
    prob = hist.astype(np.float64) / total

    # Filter out zero probabilities to avoid log2(0)
    prob_non_zero = prob[prob > 0]
    entropy_val = -np.sum(prob_non_zero * np.log2(prob_non_zero))

    return float(entropy_val)


def calculate_contrast_roi(image: np.ndarray) -> float:
    """
    Computes ROI-based Michelson/Weber contrast:
        C_ROI = |μ_f - μ_b| / (μ_b + eps)
    """
    _, _, mu_f, mu_b, _ = extract_roi_masks(image)
    return float(abs(mu_f - mu_b) / (mu_b + 1e-8))


def calculate_cii_roi(original_img: np.ndarray, enhanced_img: np.ndarray) -> float:
    """
    Computes ROI-based Contrast Improvement Index:
        CII_ROI = C_processed / (C_original + eps)
    where C = |μ_f - μ_b| / μ_b
    """
    c_orig = calculate_contrast_roi(original_img)
    c_enh = calculate_contrast_roi(enhanced_img)
    return float(c_enh / (c_orig + 1e-8))


def calculate_image_contrast(image: np.ndarray, window_size: int = 16) -> float:
    """
    Computes local patch-based Michelson/Weber contrast for a mammogram image.
    
    Divides the image into non-overlapping windows and computes:
        C_patch = (I_max - I_min) / (I_max + I_min + eps)
    and returns the mean contrast across all windows.

    Args:
        image (np.ndarray): 2D grayscale image.
        window_size (int): Tile window dimension for local contrast evaluation (default: 16).

    Returns:
        float: Average local contrast value in [0.0, 1.0].
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input must be a numpy.ndarray, got {type(image)}")
    if image.size == 0:
        raise ValueError("Input image is empty.")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    img_float = image.astype(np.float64)
    if img_float.max() > 1.0:
        img_float = img_float / 255.0

    H, W = img_float.shape[:2]
    contrasts = []
    eps = 1e-6

    # Extract non-overlapping patches
    for y in range(0, H - window_size + 1, window_size):
        for x in range(0, W - window_size + 1, window_size):
            patch = img_float[y:y + window_size, x:x + window_size]
            p_min = patch.min()
            p_max = patch.max()
            denominator = p_max + p_min + eps
            c = (p_max - p_min) / denominator
            contrasts.append(c)

    if not contrasts:
        # Fallback to whole-image Michelson contrast if image is smaller than window
        c_global = (img_float.max() - img_float.min()) / (img_float.max() + img_float.min() + eps)
        return float(c_global)

    return float(np.mean(contrasts))


def calculate_contrast_improvement_index(
    original_img: np.ndarray,
    enhanced_img: np.ndarray,
    window_size: int = 16
) -> float:
    """
    Computes Patch-based Contrast Improvement Index (CII_Patch) between original and enhanced images.
    
    Formula:
        CII_Patch = C_patch,enhanced / (C_patch,original + eps)
    where C_patch is the local patch-based contrast.
    """
    c_orig = calculate_image_contrast(original_img, window_size=window_size)
    c_enh = calculate_image_contrast(enhanced_img, window_size=window_size)

    eps = 1e-8
    cii = c_enh / (c_orig + eps)
    return float(cii)


def calculate_cii_patch(
    original_img: np.ndarray,
    enhanced_img: np.ndarray,
    window_size: int = 16
) -> float:
    """Alias for calculate_contrast_improvement_index."""
    return calculate_contrast_improvement_index(original_img, enhanced_img, window_size=window_size)


def calculate_snr(image: np.ndarray) -> float:
    """
    Computes Signal-to-Noise Ratio (SNR):
        SNR = μ_ROI / σ_background
    """
    _, _, mu_f, _, sigma_b = extract_roi_masks(image)
    return float(mu_f / (sigma_b + 1e-8))


def calculate_cnr(image: np.ndarray) -> float:
    """
    Computes Contrast-to-Noise Ratio (CNR):
        CNR = |μ_ROI - μ_background| / σ_background
    """
    _, _, mu_f, mu_b, sigma_b = extract_roi_masks(image)
    return float(abs(mu_f - mu_b) / (sigma_b + 1e-8))


def evaluate_enhancement_metrics(
    original_img: np.ndarray,
    enhanced_img: np.ndarray,
    window_size: int = 16
) -> Dict[str, Union[float, str]]:
    """
    Computes comprehensive image quality & contrast enhancement metrics:
    - CII_ROI: ROI-based Contrast Improvement Index
    - CII_Patch: Patch-based Contrast Improvement Index
    - PSNR: Peak Signal-to-Noise Ratio
    - SSIM: Structural Similarity Index Measure
    - SNR: Signal-to-Noise Ratio (Original, Enhanced, Change %)
    - CNR: Contrast-to-Noise Ratio (Original, Enhanced, Change %)
    - Shannon Entropy (Original, Enhanced, Change %)
    - Status indicators (Improved / Degraded)

    Args:
        original_img (np.ndarray): Original/reference 2D image (uint8 or float).
        enhanced_img (np.ndarray): Enhanced 2D image (uint8 or float).
        window_size (int): Tile window size for CII_Patch evaluation (default: 16).

    Returns:
        Dict[str, Union[float, str]]: Dictionary of computed metric values and status.
    """
    if original_img.dtype != np.uint8:
        orig_u8 = np.clip(original_img * 255.0 if original_img.max() <= 1.0 else original_img, 0, 255).astype(np.uint8)
    else:
        orig_u8 = original_img

    if enhanced_img.dtype != np.uint8:
        enh_u8 = np.clip(enhanced_img * 255.0 if enhanced_img.max() <= 1.0 else enhanced_img, 0, 255).astype(np.uint8)
    else:
        enh_u8 = enhanced_img

    if orig_u8.ndim == 3 and orig_u8.shape[2] == 1:
        orig_u8 = orig_u8.squeeze(axis=2)
    if enh_u8.ndim == 3 and enh_u8.shape[2] == 1:
        enh_u8 = enh_u8.squeeze(axis=2)

    # 1. PSNR & SSIM
    psnr_val = float(peak_signal_noise_ratio(orig_u8, enh_u8, data_range=255))
    ssim_val = float(structural_similarity(orig_u8, enh_u8, data_range=255))

    # 2. Shannon Entropy
    orig_entropy = calculate_entropy(orig_u8)
    enh_entropy = calculate_entropy(enh_u8)
    entropy_change_pct = float((enh_entropy - orig_entropy) / (orig_entropy + 1e-8) * 100.0)

    # 3. Dual CII (ROI-based & Patch-based)
    cii_roi_val = calculate_cii_roi(orig_u8, enh_u8)
    cii_patch_val = calculate_cii_patch(orig_u8, enh_u8, window_size=window_size)

    # 4. SNR & CNR
    snr_orig = calculate_snr(orig_u8)
    snr_enh = calculate_snr(enh_u8)
    snr_change_pct = float((snr_enh - snr_orig) / (snr_orig + 1e-8) * 100.0)

    cnr_orig = calculate_cnr(orig_u8)
    cnr_enh = calculate_cnr(enh_u8)
    cnr_change_pct = float((cnr_enh - cnr_orig) / (cnr_orig + 1e-8) * 100.0)

    # 5. Laplacian Variance
    lap_orig = float(cv2.Laplacian(orig_u8, cv2.CV_64F).var())
    lap_enh = float(cv2.Laplacian(enh_u8, cv2.CV_64F).var())

    return {
        'PSNR': psnr_val,
        'SSIM': ssim_val,
        'CII_ROI': cii_roi_val,
        'CII_Patch': cii_patch_val,
        'Original_Entropy': orig_entropy,
        'Enhanced_Entropy': enh_entropy,
        'Entropy_Change_Pct': entropy_change_pct,
        'SNR_Original': snr_orig,
        'SNR_Enhanced': snr_enh,
        'SNR_Change_Pct': snr_change_pct,
        'CNR_Original': cnr_orig,
        'CNR_Enhanced': cnr_enh,
        'CNR_Change_Pct': cnr_change_pct,
        'Laplacian_Variance_Original': lap_orig,
        'Laplacian_Variance_Enhanced': lap_enh,
        'CII_ROI_Status': 'Improved' if cii_roi_val > 1.0 else 'Degraded',
        'CII_Patch_Status': 'Improved' if cii_patch_val > 1.0 else 'Degraded',
        'SNR_Status': 'Improved' if snr_change_pct > 0 else 'Degraded',
        'CNR_Status': 'Improved' if cnr_change_pct > 0 else 'Degraded',
        'Entropy_Status': 'Improved' if entropy_change_pct > 0 else 'Degraded'
    }


# =====================================================================
# Main Mammogram Enhancement Pipeline Class
# =====================================================================

class MammogramEnhancer:
    """
    Production-ready image processor for medical mammography ROI crops.
    
    Provides contrast enhancement via Histogram Equalization (HE), CLAHE,
    and high-frequency edge sharpening via Unsharp Masking (UM) for single
    images or image batches, along with quantitative benchmarking utilities.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        blur_kernel_size: Tuple[int, int] = (5, 5),
        blur_sigma: float = 1.0,
        sharpen_amount: float = 1.2,
        window_size: int = 16
    ) -> None:
        """
        Initialize the mammography contrast and sharpening pipeline parameters.

        Args:
            clip_limit (float): Threshold for contrast limiting in CLAHE (default: 2.0).
            tile_grid_size (Tuple[int, int]): Grid dimensions for CLAHE contextual tiles (default: (8, 8)).
            blur_kernel_size (Tuple[int, int]): Gaussian kernel size for Unsharp Masking (default: (5, 5)).
            blur_sigma (float): Gaussian standard deviation for Unsharp Masking (default: 1.0).
            sharpen_amount (float): Scaling factor (alpha) for high-pass detail signal (default: 1.2).
            window_size (int): Tile window size for CII evaluation (default: 16).
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
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.sharpen_amount = sharpen_amount
        self.window_size = window_size

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

    def apply_he(self, image: np.ndarray) -> np.ndarray:
        """
        Baseline Method: Apply Global Histogram Equalization (HE).
        
        Used purely to demonstrate global contrast spreading and highlight why CLAHE
        is superior (avoids noise amplification in background and fatty tissue).
        """
        img_validated = self._validate_image(image)
        return cv2.equalizeHist(img_validated)

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Primary Method: Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Divides image into contextual tiles and locally equalizes contrast with clipping
        to avoid noise over-amplification in background/fatty tissue.
        """
        img_validated = self._validate_image(image)
        return self._clahe.apply(img_validated)

    def apply_unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpening Method: Apply Unsharp Masking (UM) to sharpen mass margins and microcalcifications.
        
        Formula:
            HighPass = Original - GaussianBlur(Original, ksize, sigma)
            Output = Original + alpha * HighPass
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

    def apply_clahe_unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Final Pipeline Method: Apply CLAHE followed by Unsharp Masking.
        
        Combines local contrast enhancement with edge definition sharpening.
        """
        clahe_img = self.apply_clahe(image)
        return self.apply_unsharp_mask(clahe_img)

    def process_image(self, image: np.ndarray, method: str = "clahe_unsharp_mask") -> np.ndarray:
        """
        Process a single mammogram ROI image using the selected method.

        Args:
            image (np.ndarray): 2D uint8 image (H, W) or (H, W, 1).
            method (str): Technique to apply ('he', 'clahe', 'unsharp_mask', 'clahe_unsharp_mask').

        Returns:
            np.ndarray: Processed uint8 2D image.
        """
        valid_methods = ['he', 'clahe', 'unsharp_mask', 'clahe_unsharp_mask']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")

        if method == 'he':
            return self.apply_he(image)
        elif method == 'clahe':
            return self.apply_clahe(image)
        elif method == 'unsharp_mask':
            return self.apply_unsharp_mask(image)
        elif method == 'clahe_unsharp_mask':
            return self.apply_clahe_unsharp_mask(image)

    def process_batch(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        method: str = "clahe_unsharp_mask"
    ) -> np.ndarray:
        """
        Process a batch of images shaped (N, H, W), (N, H, W, 1), or a list of 2D arrays.
        """
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                processed_list = [self.process_image(img, method=method) for img in images]
                return np.stack(processed_list, axis=0)
            elif images.ndim == 4 and images.shape[3] == 1:
                processed_list = [self.process_image(img.squeeze(axis=2), method=method) for img in images]
                return np.stack(processed_list, axis=0)
            elif images.ndim == 2:
                return self.process_image(images, method=method)
            else:
                raise ValueError(f"Unsupported batch array shape: {images.shape}")
        elif isinstance(images, (list, tuple)):
            if len(images) == 0:
                raise ValueError("Input image list is empty.")
            processed_list = [self.process_image(img, method=method) for img in images]
            return np.stack(processed_list, axis=0)
        else:
            raise TypeError(f"Unsupported batch type: {type(images)}")

    def compare_all_methods(
        self,
        image: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        """
        Evaluates all Section 3 techniques against the original reference image:
        1. Original (Reference)
        2. Histogram Equalization (HE Baseline)
        3. CLAHE (Local Adaptive)
        4. Unsharp Masking (UM Sharpening)
        5. Combined CLAHE + UM (Final Pipeline)

        Returns:
            Tuple[Dict[str, np.ndarray], pd.DataFrame]:
                - Dict mapping method names to enhanced image arrays.
                - DataFrame comparing quantitative evaluation metrics (PSNR, SSIM, Entropy, CII, Laplacian Var).
        """
        img_validated = self._validate_image(image)

        he_img = self.apply_he(img_validated)
        clahe_img = self.apply_clahe(img_validated)
        um_img = self.apply_unsharp_mask(img_validated)
        combined_img = self.apply_clahe_unsharp_mask(img_validated)

        methods_dict = {
            'Original': img_validated,
            'HE': he_img,
            'CLAHE': clahe_img,
            'Unsharp_Mask': um_img,
            'CLAHE_plus_UM': combined_img
        }

        records = []
        for name, proc_img in methods_dict.items():
            metrics = evaluate_enhancement_metrics(img_validated, proc_img, window_size=self.window_size)
            row = {'Method': name, **metrics}
            records.append(row)

        df_comparison = pd.DataFrame(records)
        return methods_dict, df_comparison

    def parameter_sweep(
        self,
        image: np.ndarray,
        clip_limits: Optional[List[float]] = None,
        tile_grids: Optional[List[Tuple[int, int]]] = None,
        blur_sigmas: Optional[List[float]] = None,
        sharpen_amounts: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Performs a systematic parameter sweep for CLAHE and Unsharp Masking tuning.
        Supports Day 4-7 hyperparameter optimization schedule.

        Args:
            image (np.ndarray): 2D uint8 mammogram ROI.
            clip_limits (Optional[List[float]]): List of CLAHE clip limits (e.g. [1.0, 2.0, 3.0, 4.0]).
            tile_grids (Optional[List[Tuple[int, int]]]): List of grid sizes (e.g. [(4, 4), (8, 8), (16, 16)]).
            blur_sigmas (Optional[List[float]]): List of Gaussian blur sigmas (e.g. [0.5, 1.0, 1.5]).
            sharpen_amounts (Optional[List[float]]): List of sharpen alpha multipliers (e.g. [0.8, 1.2, 1.6]).

        Returns:
            pd.DataFrame: Table of evaluation metrics for all parameter configurations.
        """
        img_validated = self._validate_image(image)

        if clip_limits is None:
            clip_limits = [1.0, 2.0, 3.0, 4.0]
        if tile_grids is None:
            tile_grids = [(4, 4), (8, 8), (16, 16)]
        if blur_sigmas is None:
            blur_sigmas = [0.5, 1.0, 1.5]
        if sharpen_amounts is None:
            sharpen_amounts = [0.8, 1.2, 1.6]

        records = []
        for clip in clip_limits:
            for grid in tile_grids:
                for sigma in blur_sigmas:
                    for amount in sharpen_amounts:
                        enhancer = MammogramEnhancer(
                            clip_limit=clip,
                            tile_grid_size=grid,
                            blur_kernel_size=self.blur_kernel_size,
                            blur_sigma=sigma,
                            sharpen_amount=amount,
                            window_size=self.window_size
                        )
                        res = enhancer.apply_clahe_unsharp_mask(img_validated)
                        metrics = evaluate_enhancement_metrics(img_validated, res, window_size=self.window_size)
                        record = {
                            'Clip_Limit': clip,
                            'Tile_Grid': f"{grid[0]}x{grid[1]}",
                            'Blur_Sigma': sigma,
                            'Sharpen_Amount': amount,
                            **metrics
                        }
                        records.append(record)

        return pd.DataFrame(records)

    def visualize_enhancement(
        self,
        original_img: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 5)
    ) -> Tuple[Dict[str, np.ndarray], plt.Figure]:
        """
        Generates a 5-panel visual comparison displaying all Section 3 methods:
        1. Original Denoised
        2. Baseline Histogram Equalization (HE)
        3. CLAHE (Local Adaptive Contrast)
        4. Standalone Unsharp Masking (UM)
        5. Combined CLAHE + Unsharp Masking (Final Pipeline)

        Args:
            original_img (np.ndarray): Input 2D uint8 mammogram ROI.
            save_path (Optional[str]): File path to save output figure.
            figsize (Tuple[int, int]): Size of figure (default: (20, 5)).

        Returns:
            Tuple[Dict[str, np.ndarray], plt.Figure]:
                - Dictionary containing all enhanced images.
                - Matplotlib Figure handle.
        """
        methods_dict, df_metrics = self.compare_all_methods(original_img)

        fig, axes = plt.subplots(1, 5, figsize=figsize)

        titles = [
            "1. Original (Denoised)",
            "2. Global HE (Baseline)",
            f"3. CLAHE (clip={self.clip_limit})",
            f"4. Unsharp Mask (α={self.sharpen_amount})",
            f"5. Combined CLAHE+UM (Final)"
        ]

        keys = ['Original', 'HE', 'CLAHE', 'Unsharp_Mask', 'CLAHE_plus_UM']

        for idx, key in enumerate(keys):
            img = methods_dict[key]
            axes[idx].imshow(img, cmap='gray', vmin=0, vmax=255)
            row = df_metrics[df_metrics['Method'] == key].iloc[0]
            
            if key == 'Original':
                subtitle = f"Entropy: {row['Original_Entropy']:.2f} bits"
            else:
                subtitle = f"Entropy: {row['Enhanced_Entropy']:.2f} | CII: {row['CII']:.2f}\nSSIM: {row['SSIM']:.3f} | PSNR: {row['PSNR']:.1f}dB"

            axes[idx].set_title(f"{titles[idx]}\n{subtitle}", fontsize=10, fontweight='bold')
            axes[idx].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        return methods_dict, fig


# =====================================================================
# Functional Wrapper Interface
# =====================================================================

def process_mammogram_roi(
    image: np.ndarray,
    method: str = "clahe_unsharp_mask",
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    blur_kernel_size: Tuple[int, int] = (5, 5),
    blur_sigma: float = 1.0,
    sharpen_amount: float = 1.2
) -> np.ndarray:
    """
    Convenience functional interface to process single images or batches with Stage 5 enhancement.
    """
    enhancer = MammogramEnhancer(
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
        sharpen_amount=sharpen_amount
    )
    if image.ndim in (3, 4) and (image.shape[0] > 1 or image.ndim == 4):
        return enhancer.process_batch(image, method=method)
    return enhancer.process_image(image, method=method)
