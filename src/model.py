import cv2
import numpy as np
from skimage.util import random_noise
from skimage.restoration import wiener
from scipy.ndimage import gaussian_filter as gf_ndimage, uniform_filter

# --- Custom Anscombe Transform Functions ---
def anscombe(image, constant=3/8):
    """Applies the Anscombe transform for variance stabilization of Poisson data."""
    image_float = image.astype(np.float64)
    return 2.0 * np.sqrt(image_float + constant)

def inverse_anscombe(transformed_image, constant=3/8):
    """Applies the inverse Anscombe transform."""
    transformed_image_float = transformed_image.astype(np.float64)
    return np.maximum(0.0, (transformed_image_float / 2.0)**2 - constant)

# --- Noise Injection Functions ---
def add_mixed_poisson_gaussian_noise(img, gaussian_var=0.01):
    """Applies Poisson noise followed by Gaussian noise."""
    poisson_noisy_img = random_noise(img, mode='poisson')
    mixed_noisy_img = random_noise(poisson_noisy_img, mode='gaussian', var=gaussian_var)
    return mixed_noisy_img

def add_noise(image, noise_type, config):
    """
    Applies specified noise to an image based on configurations.
    """
    noise_cfg = config.get('noise', {})
    if noise_type == 'gaussian':
        var = noise_cfg.get('gaussian', {}).get('var', 0.01)
        return random_noise(image, mode='gaussian', var=var)
    elif noise_type == 's&p':
        amount = noise_cfg.get('sp', {}).get('amount', 0.05)
        return random_noise(image, mode='s&p', amount=amount)
    elif noise_type == 'speckle':
        var = noise_cfg.get('speckle', {}).get('var', 0.04)
        return random_noise(image, mode='speckle', var=var)
    elif noise_type == 'poisson':
        return random_noise(image, mode='poisson')
    elif noise_type == 'mixed_poisson_gaussian':
        g_var = noise_cfg.get('mixed_poisson_gaussian', {}).get('gaussian_var', 0.01)
        return add_mixed_poisson_gaussian_noise(image, gaussian_var=g_var)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

# --- Denoising Filter Implementations ---
def wiener_filter(img, balance=0.1):
    """Applies scikit-image's Wiener filter implementation."""
    dummy_psf = gf_ndimage(np.zeros((3,3)), 1, output=None, mode='constant')
    dummy_psf[1,1] = 1
    dummy_psf /= dummy_psf.sum()
    denoised_img = wiener(img, dummy_psf, balance=balance)
    return denoised_img

def anscombe_wiener_denoising(img, balance=0.1):
    """Applies Anscombe transform, then Wiener filter, then inverse Anscombe transform."""
    transformed_img = anscombe(img)
    denoised_transformed_img = wiener_filter(transformed_img, balance=balance)
    denoised_img = inverse_anscombe(denoised_transformed_img)
    return np.clip(denoised_img, 0, 1)

def adaptive_median_filter(img, s_max=7):
    """Applies an adaptive median filter to a grayscale image (float, [0,1])."""
    img_uint8 = (img * 255).astype(np.uint8)
    H, W = img_uint8.shape
    out = np.copy(img_uint8)

    for i in range(H):
        for j in range(W):
            s = 3 # initial window size
            while s <= s_max:
                half_s = s // 2
                r_min, r_max = max(0, i - half_s), min(H, i + half_s + 1)
                c_min, c_max = max(0, j - half_s), min(W, j + half_s + 1)
                window = img_uint8[r_min:r_max, c_min:c_max]

                if window.size == 0:
                    s += 2
                    continue

                med = np.median(window)
                z_min = np.min(window)
                z_max = np.max(window)

                pixel_val = img_uint8[i, j]

                if z_min < med < z_max:
                    if z_min < pixel_val < z_max:
                        out[i, j] = pixel_val
                    else:
                        out[i, j] = med
                    break
                else:
                    s += 2
                    if s > s_max:
                        out[i, j] = med
                        break
    return out.astype(np.float64) / 255.0

def kuan_filter(img, win_size=5, noise_var_estimate=0.04):
    """Applies the Kuan filter for speckle noise reduction."""
    local_mean = uniform_filter(img, size=win_size)
    local_sq_mean = uniform_filter(img**2, size=win_size)
    local_variance = local_sq_mean - local_mean**2

    # Clamp local_variance to avoid division by zero
    local_variance[local_variance < 1e-6] = 1e-6

    # Kuan filter coefficient (B)
    B = 1 - (noise_var_estimate / local_variance)
    B[B < 0] = 0

    denoised_img = local_mean + B * (img - local_mean)
    return np.clip(denoised_img, 0, 1)

def apply_denoising(noisy_image, method, config):
    """
    Applies specified denoising method using configuration settings.
    """
    denoise_cfg = config.get('denoising', {})
    
    if method == 'median':
        ksize = denoise_cfg.get('median', {}).get('kernel_size', 5)
        # OpenCV medianBlur expects integer uint8 image
        img_uint8 = (noisy_image * 255).astype(np.uint8)
        denoised = cv2.medianBlur(img_uint8, ksize).astype(np.float64) / 255.0
        return np.clip(denoised, 0, 1)
        
    elif method == 'gaussian':
        ksize_tuple = tuple(denoise_cfg.get('gaussian', {}).get('kernel_size', [5, 5]))
        sigma = denoise_cfg.get('gaussian', {}).get('sigma', 0)
        img_uint8 = (noisy_image * 255).astype(np.uint8)
        denoised = cv2.GaussianBlur(img_uint8, ksize_tuple, sigma).astype(np.float64) / 255.0
        return np.clip(denoised, 0, 1)
        
    elif method == 'wiener':
        balance = denoise_cfg.get('wiener', {}).get('balance', 0.1)
        return np.clip(wiener_filter(noisy_image, balance=balance), 0, 1)
        
    elif method == 'bilateral':
        d = denoise_cfg.get('bilateral', {}).get('d', 9)
        sc = denoise_cfg.get('bilateral', {}).get('sigma_color', 75)
        ss = denoise_cfg.get('bilateral', {}).get('sigma_space', 75)
        img_uint8 = (noisy_image * 255).astype(np.uint8)
        denoised = cv2.bilateralFilter(img_uint8, d, sc, ss).astype(np.float64) / 255.0
        return np.clip(denoised, 0, 1)
        
    elif method == 'non_local_means':
        h = denoise_cfg.get('non_local_means', {}).get('h', 10)
        tw = denoise_cfg.get('non_local_means', {}).get('template_window_size', 7)
        sw = denoise_cfg.get('non_local_means', {}).get('search_window_size', 21)
        img_uint8 = (noisy_image * 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(img_uint8, None, h, tw, sw).astype(np.float64) / 255.0
        return np.clip(denoised, 0, 1)
        
    elif method == 'anscombe_wiener':
        balance = denoise_cfg.get('wiener', {}).get('balance', 0.1)
        return np.clip(anscombe_wiener_denoising(noisy_image, balance=balance), 0, 1)
        
    elif method == 'adaptive_median':
        s_max = denoise_cfg.get('adaptive_median', {}).get('s_max', 7)
        return np.clip(adaptive_median_filter(noisy_image, s_max=s_max), 0, 1)
        
    elif method == 'kuan':
        # Default parameter fallbacks
        return np.clip(kuan_filter(noisy_image), 0, 1)
        
    else:
        raise ValueError(f"Unsupported denoising method: {method}")
