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
    Reads an image, converts it to grayscale, resizes it, and normalizes pixel values.
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
    Saves a float image [0,1] as uint8 PNG.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        image_array = np.squeeze(image_array)
        image_array = np.clip(image_array, 0.0, 1.0)

        img_to_save = (image_array * 255).astype(np.uint8)

        cv2.imwrite(output_path, img_to_save)

    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")

def calculate_metrics(original_img, noisy_img, denoised_img):
    """
    Calculates image quality metrics (Mean, Median, Std Dev, MSE, PSNR, SSIM).
    """
    metrics = {}
    
    # Metrics for Original Image (characteristics)
    metrics['Original_Mean'] = np.mean(original_img)
    metrics['Original_Median'] = np.median(original_img)
    metrics['Original_StdDev'] = np.std(original_img)

    # MSE: Mean Squared Error between two images
    metrics['MSE_Noisy_vs_Original'] = mean_squared_error(original_img, noisy_img)
    metrics['MSE_Denoised_vs_Original'] = mean_squared_error(original_img, denoised_img)

    # PSNR: Peak Signal-to-Noise Ratio (data_range=1 for float images)
    metrics['PSNR_Noisy_vs_Original'] = peak_signal_noise_ratio(original_img, noisy_img, data_range=1.0)
    metrics['PSNR_Denoised_vs_Original'] = peak_signal_noise_ratio(original_img, denoised_img, data_range=1.0)

    # SSIM: Structural Similarity Index
    metrics['SSIM_Noisy_vs_Original'] = structural_similarity(original_img, noisy_img, data_range=1.0)
    metrics['SSIM_Denoised_vs_Original'] = structural_similarity(original_img, denoised_img, data_range=1.0)

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
