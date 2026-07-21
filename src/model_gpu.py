import torch
import torch.nn.functional as F
import numpy as np

# --- GPU Batch Anscombe Transform Functions ---
def anscombe_gpu(image_tensor, constant=3/8):
    """
    Applies the Anscombe transform for variance stabilization of Poisson data on GPU.
    Supports single images (1, 1, H, W) and batched tensors (B, 1, H, W).
    """
    return 2.0 * torch.sqrt(image_tensor + constant)

def inverse_anscombe_gpu(transformed_image_tensor, constant=3/8):
    """
    Applies the inverse Anscombe transform on GPU.
    Supports single images (1, 1, H, W) and batched tensors (B, 1, H, W).
    """
    return torch.clamp((transformed_image_tensor / 2.0)**2 - constant, min=0.0)

# --- GPU Batch Noise Injection Functions ---
def add_noise_gpu(image_tensor, noise_type, config, device):
    """
    Applies specified noise model to a batch of PyTorch tensors (B, 1, H, W) on GPU in parallel.
    """
    if image_tensor.ndim == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    elif image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(1)

    noise_cfg = config.get('noise', {})
    
    if noise_type == 'gaussian':
        var = noise_cfg.get('gaussian', {}).get('var', 0.01)
        std = var ** 0.5
        noisy = image_tensor + torch.randn_like(image_tensor) * std
        return torch.clamp(noisy, 0.0, 1.0)
        
    elif noise_type == 's&p':
        amount = noise_cfg.get('sp', {}).get('amount', 0.05)
        noisy = image_tensor.clone()
        random_matrix = torch.rand_like(image_tensor)
        # Salt (1.0)
        noisy[random_matrix < (amount / 2.0)] = 1.0
        # Pepper (0.0)
        noisy[random_matrix > (1.0 - amount / 2.0)] = 0.0
        return noisy
        
    elif noise_type == 'speckle':
        var = noise_cfg.get('speckle', {}).get('var', 0.04)
        std = var ** 0.5
        noisy = image_tensor + image_tensor * torch.randn_like(image_tensor) * std
        return torch.clamp(noisy, 0.0, 1.0)
        
    elif noise_type == 'poisson':
        # Scale to discrete photon counts (default 256 levels), apply Poisson distribution sampling
        vals = 256.0
        noisy = torch.poisson(image_tensor * vals) / vals
        return torch.clamp(noisy, 0.0, 1.0)
        
    elif noise_type == 'mixed_poisson_gaussian':
        g_var = noise_cfg.get('mixed_poisson_gaussian', {}).get('gaussian_var', 0.01)
        vals = 256.0
        poisson_noisy = torch.poisson(image_tensor * vals) / vals
        std = g_var ** 0.5
        noisy = poisson_noisy + torch.randn_like(poisson_noisy) * std
        return torch.clamp(noisy, 0.0, 1.0)
        
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

# --- GPU Batch Denoising Filter Implementations ---
def median_filter_gpu(img, ksize=5):
    """
    Applies 2D spatial median filter on GPU for a batch of images (B, 1, H, W).
    Uses sliding-window tensor unfolding for fast parallel median evaluation across batch.
    """
    squeeze_needed = False
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
        squeeze_needed = True
    elif img.ndim == 3:
        img = img.unsqueeze(1)

    B, C, H, W = img.shape
    pad = ksize // 2
    img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')
    
    # Unfold sliding windows into patches (B, C, H, W, ksize, ksize)
    patches = img_padded.unfold(2, ksize, 1).unfold(3, ksize, 1)
    patches = patches.contiguous().view(B, C, H, W, -1)
    median_val, _ = patches.median(dim=-1)
    
    return median_val.squeeze(0).squeeze(0) if squeeze_needed else median_val

def gaussian_filter_gpu(img, ksize=(5, 5), sigma=1.0):
    """
    Applies 2D Gaussian convolution on GPU for a batch of images (B, 1, H, W).
    """
    squeeze_needed = False
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
        squeeze_needed = True
    elif img.ndim == 3:
        img = img.unsqueeze(1)

    kx = torch.arange(-(ksize[0]//2), ksize[0]//2 + 1, dtype=torch.float32, device=img.device)
    ky = torch.arange(-(ksize[1]//2), ksize[1]//2 + 1, dtype=torch.float32, device=img.device)
    if sigma <= 0:
        sigma = 0.3 * ((ksize[0] - 1) * 0.5 - 1) + 0.8
    gx = torch.exp(-kx**2 / (2 * sigma**2))
    gy = torch.exp(-ky**2 / (2 * sigma**2))
    kernel = (gx.unsqueeze(1) * gy.unsqueeze(0))
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    pad_h = ksize[0] // 2
    pad_w = ksize[1] // 2
    img_padded = F.pad(img, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    denoised = F.conv2d(img_padded, kernel)
    
    return denoised.squeeze(0).squeeze(0) if squeeze_needed else denoised

def wiener_filter_gpu(img, mysize=3, noise=None):
    """
    Applies local statistics Wiener filter on GPU for a batch of images (B, 1, H, W).
    """
    squeeze_needed = False
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
        squeeze_needed = True
    elif img.ndim == 3:
        img = img.unsqueeze(1)

    pad = mysize // 2
    img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')
    local_mean = F.avg_pool2d(img_padded, kernel_size=mysize, stride=1)
    local_mean_sq = F.avg_pool2d(img_padded**2, kernel_size=mysize, stride=1)
    local_var = local_mean_sq - local_mean**2
    
    if noise is None:
        noise = torch.mean(local_var, dim=(-2, -1), keepdim=True)
        
    res = local_mean + (torch.clamp(local_var - noise, min=0.0) / torch.clamp(local_var, min=1e-8)) * (img - local_mean)
    return res.squeeze(0).squeeze(0) if squeeze_needed else res

def anscombe_wiener_denoising_gpu(img, balance=0.1):
    """
    Applies Anscombe transformation, Wiener filtering, and Inverse Anscombe mapping on GPU.
    """
    transformed_img = anscombe_gpu(img)
    denoised_transformed_img = wiener_filter_gpu(transformed_img, mysize=3, noise=balance)
    denoised_img = inverse_anscombe_gpu(denoised_transformed_img)
    return torch.clamp(denoised_img, 0.0, 1.0)

def bilateral_filter_gpu(img, d=9, sigma_color=75.0, sigma_space=75.0):
    """
    Applies 2D Bilateral filter on GPU for a batch of images (B, 1, H, W).
    Vectorized patch extraction and range/spatial exponential kernel weighting.
    """
    squeeze_needed = False
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
        squeeze_needed = True
    elif img.ndim == 3:
        img = img.unsqueeze(1)

    B, C, H, W = img.shape
    pad = d // 2
    img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')
    
    x = torch.arange(-pad, pad + 1, dtype=torch.float32, device=img.device)
    y = torch.arange(-pad, pad + 1, dtype=torch.float32, device=img.device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    spatial_w = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma_space**2)).view(1, 1, 1, 1, -1)
    
    patches = img_padded.unfold(2, d, 1).unfold(3, d, 1).contiguous().view(B, C, H, W, -1)
    center = img.unsqueeze(-1)
    
    color_w = torch.exp(-(patches - center)**2 / (2 * (sigma_color / 255.0)**2))
    total_w = spatial_w * color_w
    total_w_sum = total_w.sum(dim=-1, keepdim=True)
    
    denoised = (patches * total_w).sum(dim=-1, keepdim=True) / torch.clamp(total_w_sum, min=1e-8)
    denoised = denoised.squeeze(-1)
    
    return denoised.squeeze(0).squeeze(0) if squeeze_needed else denoised

def non_local_means_gpu(img, h=10, template_window_size=7, search_window_size=21):
    """
    Applies Non-Local Means filter on GPU for a batch of images (B, 1, H, W).
    Fully parallelized patch distance evaluation using box-filter average pooling across search window.
    """
    squeeze_needed = False
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
        squeeze_needed = True
    elif img.ndim == 3:
        img = img.unsqueeze(1)

    B, C, H, W = img.shape
    t_pad = template_window_size // 2
    s_pad = search_window_size // 2
    h_scaled = h / 255.0
    h2 = h_scaled * h_scaled
    
    total_pad = t_pad + s_pad
    img_padded = F.pad(img, (total_pad, total_pad, total_pad, total_pad), mode='reflect')
    
    denoised = torch.zeros_like(img)
    weight_sum = torch.zeros_like(img)
    
    for dy in range(-s_pad, s_pad + 1):
        for dx in range(-s_pad, s_pad + 1):
            shifted = img_padded[:, :, total_pad + dy : total_pad + dy + H, total_pad + dx : total_pad + dx + W]
            diff = (img - shifted) ** 2
            pad_diff = F.pad(diff, (t_pad, t_pad, t_pad, t_pad), mode='reflect')
            patch_dist = F.avg_pool2d(pad_diff, kernel_size=template_window_size, stride=1) * (template_window_size**2)
            
            weight = torch.exp(-patch_dist / h2)
            denoised += weight * shifted
            weight_sum += weight
            
    denoised = denoised / torch.clamp(weight_sum, min=1e-8)
    return denoised.squeeze(0).squeeze(0) if squeeze_needed else denoised

def adaptive_median_filter_gpu(img, s_max=7):
    """
    Applies Adaptive Median filter on GPU for a batch of images (B, 1, H, W).
    Iteratively increases window size for impulse noise pixels while preserving uncorrupted structures.
    """
    squeeze_needed = False
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
        squeeze_needed = True
    elif img.ndim == 3:
        img = img.unsqueeze(1)
    
    B, C, H, W = img.shape
    output = img.clone()
    active_mask = torch.ones_like(img, dtype=torch.bool)
    
    for s in range(3, s_max + 1, 2):
        if not active_mask.any():
            break
        pad = s // 2
        img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')
        patches = img_padded.unfold(2, s, 1).unfold(3, s, 1)
        patches = patches.contiguous().view(B, C, H, W, -1)
        
        z_min = patches.min(dim=-1).values
        z_max = patches.max(dim=-1).values
        z_med = patches.median(dim=-1).values
        
        cond_A = (z_med > z_min) & (z_med < z_max)
        cond_B = (img > z_min) & (img < z_max)
        
        final_vals = torch.where(cond_B, img, z_med)
        apply_mask = active_mask & cond_A
        output[apply_mask] = final_vals[apply_mask]
        active_mask = active_mask & (~cond_A)
        
    if active_mask.any():
        s = s_max
        pad = s // 2
        img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')
        patches = img_padded.unfold(2, s, 1).unfold(3, s, 1)
        patches = patches.contiguous().view(B, C, H, W, -1)
        z_med = patches.median(dim=-1).values
        output[active_mask] = z_med[active_mask]
        
    return output.squeeze(0).squeeze(0) if squeeze_needed else output

def kuan_filter_gpu(img, win_size=5, noise_var_estimate=0.04):
    """
    Applies Kuan filter on GPU for a batch of images (B, 1, H, W).
    """
    squeeze_needed = False
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
        squeeze_needed = True
    elif img.ndim == 3:
        img = img.unsqueeze(1)

    pad = win_size // 2
    img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')
    local_mean = F.avg_pool2d(img_padded, kernel_size=win_size, stride=1)
    local_sq_mean = F.avg_pool2d(img_padded**2, kernel_size=win_size, stride=1)
    local_variance = local_sq_mean - local_mean**2
    local_variance = torch.clamp(local_variance, min=1e-6)
    
    B_coeff = 1.0 - (noise_var_estimate / local_variance)
    B_coeff = torch.clamp(B_coeff, min=0.0)
    
    denoised = local_mean + B_coeff * (img - local_mean)
    denoised = torch.clamp(denoised, 0.0, 1.0)
    
    return denoised.squeeze(0).squeeze(0) if squeeze_needed else denoised

def apply_denoising_gpu(noisy_image_tensor, method, config):
    """
    Applies specified denoising filter model to a batch of PyTorch tensors (B, 1, H, W) on GPU.
    """
    denoise_cfg = config.get('denoising', {})
    
    if method == 'median':
        ksize = denoise_cfg.get('median', {}).get('kernel_size', 5)
        return torch.clamp(median_filter_gpu(noisy_image_tensor, ksize), 0.0, 1.0)
        
    elif method == 'gaussian':
        ksize_tuple = tuple(denoise_cfg.get('gaussian', {}).get('kernel_size', [5, 5]))
        sigma = denoise_cfg.get('gaussian', {}).get('sigma', 0.0)
        return torch.clamp(gaussian_filter_gpu(noisy_image_tensor, ksize_tuple, sigma), 0.0, 1.0)
        
    elif method == 'wiener':
        balance = denoise_cfg.get('wiener', {}).get('balance', 0.1)
        return torch.clamp(wiener_filter_gpu(noisy_image_tensor, noise=balance), 0.0, 1.0)
        
    elif method == 'bilateral':
        d = denoise_cfg.get('bilateral', {}).get('d', 9)
        sc = denoise_cfg.get('bilateral', {}).get('sigma_color', 75.0)
        ss = denoise_cfg.get('bilateral', {}).get('sigma_space', 75.0)
        return torch.clamp(bilateral_filter_gpu(noisy_image_tensor, d, sc, ss), 0.0, 1.0)
        
    elif method == 'non_local_means':
        h = denoise_cfg.get('non_local_means', {}).get('h', 10)
        tw = denoise_cfg.get('non_local_means', {}).get('template_window_size', 7)
        sw = denoise_cfg.get('non_local_means', {}).get('search_window_size', 21)
        return torch.clamp(non_local_means_gpu(noisy_image_tensor, h, tw, sw), 0.0, 1.0)
        
    elif method == 'anscombe_wiener':
        balance = denoise_cfg.get('wiener', {}).get('balance', 0.1)
        return torch.clamp(anscombe_wiener_denoising_gpu(noisy_image_tensor, balance=balance), 0.0, 1.0)
        
    elif method == 'adaptive_median':
        s_max = denoise_cfg.get('adaptive_median', {}).get('s_max', 7)
        return torch.clamp(adaptive_median_filter_gpu(noisy_image_tensor, s_max=s_max), 0.0, 1.0)
        
    elif method == 'kuan':
        return torch.clamp(kuan_filter_gpu(noisy_image_tensor), 0.0, 1.0)
        
    else:
        raise ValueError(f"Unsupported denoising method: {method}")

# --- GPU Batch Image Quality Metrics (MSE, PSNR, SSIM) ---
def ssim_gpu_batch(img1, img2, window_size=11, sigma=1.5, data_range=1.0):
    """
    Computes Structural Similarity Index (SSIM) for a batch of images (B, 1, H, W) on GPU.
    Matches scikit-image structural_similarity implementation.
    """
    if img1.ndim == 3:
        img1 = img1.unsqueeze(1)
    if img2.ndim == 3:
        img2 = img2.unsqueeze(1)

    device = img1.device
    channel = img1.size(1)

    coords = torch.arange(window_size, dtype=torch.float32, device=device) - (window_size // 2)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(dim=(-3, -2, -1))

def calculate_metrics_gpu_batch(orig_batch, noisy_batch, denoised_batch):
    """
    Calculates batch image metrics (Mean, Median, StdDev, MSE, PSNR, SSIM) entirely on GPU.
    Returns: List of metric dicts, one per image in the batch.
    """
    if orig_batch.ndim == 3:
        orig_batch = orig_batch.unsqueeze(1)
    if noisy_batch.ndim == 3:
        noisy_batch = noisy_batch.unsqueeze(1)
    if denoised_batch.ndim == 3:
        denoised_batch = denoised_batch.unsqueeze(1)

    B = orig_batch.shape[0]

    # Original Image Characteristics
    orig_mean = orig_batch.mean(dim=(-3, -2, -1)).cpu().numpy()
    orig_std = orig_batch.std(dim=(-3, -2, -1)).cpu().numpy()
    orig_median = orig_batch.view(B, -1).median(dim=-1).values.cpu().numpy()

    # MSE Calculation
    mse_noisy = (orig_batch - noisy_batch).pow(2).mean(dim=(-3, -2, -1))
    mse_denoised = (orig_batch - denoised_batch).pow(2).mean(dim=(-3, -2, -1))

    # PSNR Calculation
    psnr_noisy = 10.0 * torch.log10(1.0 / torch.clamp(mse_noisy, min=1e-10))
    psnr_denoised = 10.0 * torch.log10(1.0 / torch.clamp(mse_denoised, min=1e-10))

    # SSIM Calculation
    ssim_noisy = ssim_gpu_batch(orig_batch, noisy_batch)
    ssim_denoised = ssim_gpu_batch(orig_batch, denoised_batch)

    mse_noisy_np = mse_noisy.cpu().numpy()
    mse_denoised_np = mse_denoised.cpu().numpy()
    psnr_noisy_np = psnr_noisy.cpu().numpy()
    psnr_denoised_np = psnr_denoised.cpu().numpy()
    ssim_noisy_np = ssim_noisy.cpu().numpy()
    ssim_denoised_np = ssim_denoised.cpu().numpy()

    metrics_list = []
    for i in range(B):
        metrics_list.append({
            'Original_Mean': float(orig_mean[i]),
            'Original_Median': float(orig_median[i]),
            'Original_StdDev': float(orig_std[i]),
            'MSE_Noisy_vs_Original': float(mse_noisy_np[i]),
            'MSE_Denoised_vs_Original': float(mse_denoised_np[i]),
            'PSNR_Noisy_vs_Original': float(psnr_noisy_np[i]),
            'PSNR_Denoised_vs_Original': float(psnr_denoised_np[i]),
            'SSIM_Noisy_vs_Original': float(ssim_noisy_np[i]),
            'SSIM_Denoised_vs_Original': float(ssim_denoised_np[i]),
        })

    return metrics_list
