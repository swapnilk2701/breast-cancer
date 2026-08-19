import os
import cv2
import yaml
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def load_config(config_path="config/config.yaml"):
    """Loads configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def read_and_preprocess_image(image_path, target_size=(256, 256)):
    """
    Reads an image, converts it to grayscale, resizes it, and normalizes pixel values to [0, 1].
    """
    try:
        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return None

        # Resize image
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        # Normalize pixel values to [0, 1]
        img = img.astype(np.float64) / 255.0

        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}. Skipping.")
        return None


def save_image(image_array, output_path):
    """
    Saves a float image [0,1] or uint8 image as uint8 PNG.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        image_array = np.squeeze(image_array)
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0.0, 1.0)
            img_to_save = (image_array * 255.0).astype(np.uint8)
        else:
            img_to_save = image_array

        cv2.imwrite(output_path, img_to_save)

    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")


def calculate_entropy(image):
    """
    Computes Shannon Entropy (information content / detail richness) in bits.
    Formula: H = - sum(p_i * log2(p_i))
    """
    img_array = np.squeeze(image)
    if img_array.dtype != np.uint8:
        img_u8 = np.clip(img_array * 255.0 if img_array.max() <= 1.0 else img_array, 0, 255).astype(np.uint8)
    else:
        img_u8 = img_array

    hist, _ = np.histogram(img_u8.ravel(), bins=256, range=(0, 256))
    prob = hist.astype(np.float64) / max(hist.sum(), 1)
    prob_non_zero = prob[prob > 0]
    return float(-np.sum(prob_non_zero * np.log2(prob_non_zero)))


def calculate_image_contrast(image, window_size=16):
    """
    Computes average local patch contrast for an image.
    C_patch = (I_max - I_min) / (I_max + I_min + 1e-6)
    """
    img_float = np.squeeze(image).astype(np.float64)
    if img_float.max() > 1.0:
        img_float = img_float / 255.0

    H, W = img_float.shape[:2]
    contrasts = []
    eps = 1e-6

    for y in range(0, H - window_size + 1, window_size):
        for x in range(0, W - window_size + 1, window_size):
            patch = img_float[y:y + window_size, x:x + window_size]
            p_min = patch.min()
            p_max = patch.max()
            c = (p_max - p_min) / (p_max + p_min + eps)
            contrasts.append(c)

    if not contrasts:
        return float((img_float.max() - img_float.min()) / (img_float.max() + img_float.min() + eps))

    return float(np.mean(contrasts))


def calculate_cii(original_img, enhanced_img, window_size=16):
    """
    Computes Contrast Improvement Index (CII) = C_enhanced / (C_original + eps).
    """
    c_orig = calculate_image_contrast(original_img, window_size=window_size)
    c_enh = calculate_image_contrast(enhanced_img, window_size=window_size)
    return float(c_enh / (c_orig + 1e-8))


def calculate_enhancement_metrics(original_img, enhanced_img, window_size=16):
    """
    Computes dedicated contrast enhancement metrics (PSNR, SSIM, Entropy, CII, Laplacian Var).
    """
    orig = np.squeeze(original_img).astype(np.float64)
    enh = np.squeeze(enhanced_img).astype(np.float64)

    if orig.max() > 1.0:
        orig = orig / 255.0
    if enh.max() > 1.0:
        enh = enh / 255.0

    orig_u8 = (np.clip(orig, 0, 1) * 255).astype(np.uint8)
    enh_u8 = (np.clip(enh, 0, 1) * 255).astype(np.uint8)

    psnr_val = peak_signal_noise_ratio(orig, enh, data_range=1.0)
    ssim_val = structural_similarity(orig, enh, data_range=1.0)
    orig_ent = calculate_entropy(orig_u8)
    enh_ent = calculate_entropy(enh_u8)
    cii_val = calculate_cii(orig, enh, window_size=window_size)
    lap_orig = float(cv2.Laplacian(orig_u8, cv2.CV_64F).var())
    lap_enh = float(cv2.Laplacian(enh_u8, cv2.CV_64F).var())

    return {
        'PSNR_Enhanced_vs_Original': float(psnr_val),
        'SSIM_Enhanced_vs_Original': float(ssim_val),
        'MSE_Enhanced_vs_Original': float(mean_squared_error(orig, enh)),
        'Original_Entropy': float(orig_ent),
        'Enhanced_Entropy': float(enh_ent),
        'Entropy_Delta': float(enh_ent - orig_ent),
        'CII': float(cii_val),
        'Laplacian_Variance_Original': lap_orig,
        'Laplacian_Variance_Enhanced': lap_enh
    }


def calculate_metrics(original_img, noisy_img, denoised_img, enhanced_img=None, window_size=16):
    """
    Calculates image quality metrics (Mean, Median, Std Dev, MSE, PSNR, SSIM, and optional Enhancement metrics).
    """
    metrics = {}

    # Metrics for Original Image (characteristics)
    metrics['Original_Mean'] = float(np.mean(original_img))
    metrics['Original_Median'] = float(np.median(original_img))
    metrics['Original_StdDev'] = float(np.std(original_img))
    metrics['Original_Entropy'] = calculate_entropy(original_img)

    # MSE: Mean Squared Error between two images
    metrics['MSE_Noisy_vs_Original'] = float(mean_squared_error(original_img, noisy_img))
    metrics['MSE_Denoised_vs_Original'] = float(mean_squared_error(original_img, denoised_img))

    # PSNR: Peak Signal-to-Noise Ratio (data_range=1 for float images)
    metrics['PSNR_Noisy_vs_Original'] = float(peak_signal_noise_ratio(original_img, noisy_img, data_range=1.0))
    metrics['PSNR_Denoised_vs_Original'] = float(peak_signal_noise_ratio(original_img, denoised_img, data_range=1.0))

    # SSIM: Structural Similarity Index
    metrics['SSIM_Noisy_vs_Original'] = float(structural_similarity(original_img, noisy_img, data_range=1.0))
    metrics['SSIM_Denoised_vs_Original'] = float(structural_similarity(original_img, denoised_img, data_range=1.0))

    # Optional Stage 5 Enhancement Metrics
    if enhanced_img is not None:
        enh_metrics = calculate_enhancement_metrics(original_img, enhanced_img, window_size=window_size)
        metrics.update(enh_metrics)

    return metrics


def display_sample_images(original_img, noisy_img, denoised_img, title_suffix="", show_plot=False, save_path=None):
    """
    Displays original, noisy, and denoised images side-by-side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(noisy_img, cmap='gray')
    axes[1].set_title(f'Noisy Image {title_suffix}')
    axes[1].axis('off')

    axes[2].imshow(denoised_img, cmap='gray')
    axes[2].set_title(f'Denoised Image {title_suffix}')
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    else:
        plt.close()


def display_sample_mse_images(original_img, noisy_img, denoised_img, title_suffix="", show_plot=False, save_path=None):
    """
    Displays Original Image, Noisy Squared Error (MSE) map, and Denoised Squared Error (MSE) map side-by-side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    noisy_se = (noisy_img - original_img) ** 2
    noisy_mse = np.mean(noisy_se)
    im1 = axes[1].imshow(noisy_se, cmap='hot')
    axes[1].set_title(f'Noisy Squared Error {title_suffix}\n(MSE: {noisy_mse:.6f})')
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    denoised_se = (denoised_img - original_img) ** 2
    denoised_mse = np.mean(denoised_se)
    im2 = axes[2].imshow(denoised_se, cmap='hot')
    axes[2].set_title(f'Denoised Squared Error {title_suffix}\n(MSE: {denoised_mse:.6f})')
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    else:
        plt.close()


def display_enhancement_comparison(
    original_img,
    he_img,
    clahe_img,
    um_img,
    combined_img,
    title_suffix="",
    save_path=None,
    show_plot=False
):
    """
    Displays 5-panel side-by-side comparison for all Stage 5 enhancement techniques:
    Original vs Global HE vs CLAHE vs Unsharp Masking vs Combined CLAHE+UM.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    panels = [
        ("Original Denoised", original_img),
        ("Global HE (Baseline)", he_img),
        ("CLAHE (Local Adaptive)", clahe_img),
        ("Unsharp Mask (Sharpening)", um_img),
        ("Combined CLAHE+UM (Final)", combined_img),
    ]

    for idx, (title, img) in enumerate(panels):
        img_arr = np.squeeze(img)
        if img_arr.max() <= 1.0:
            axes[idx].imshow(img_arr, cmap='gray', vmin=0.0, vmax=1.0)
        else:
            axes[idx].imshow(img_arr, cmap='gray', vmin=0, vmax=255)
        
        ent = calculate_entropy(img_arr)
        if idx == 0:
            axes[idx].set_title(f"{title}\nEntropy: {ent:.2f} bits", fontsize=10, fontweight='bold')
        else:
            cii = calculate_cii(original_img, img_arr)
            axes[idx].set_title(f"{title}\nEntropy: {ent:.2f} | CII: {cii:.2f}", fontsize=10, fontweight='bold')
        axes[idx].axis('off')

    plt.suptitle(f"Stage 5 Contrast Enhancement & Sharpening Comparison {title_suffix}", fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_metrics_comparison(df, metric_name, title, y_label, save_path=None, show_plot=False):
    """
    Generates comparison plots for specified metrics (e.g., PSNR, SSIM).
    """
    plt.figure(figsize=(12, 6))

    # Group by 'Noise Type' and 'Denoising Method' and calculate the mean of the metric
    plot_data = df.groupby(['Noise Type', 'Denoising Method'])[metric_name].mean().unstack()

    plot_data.plot(kind='bar', figsize=(15, 7))
    plt.title(title)
    plt.xlabel('Noise Type')
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Denoising Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()
