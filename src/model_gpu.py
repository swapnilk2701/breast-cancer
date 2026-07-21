import torch
import torch.nn.functional as F
import numpy as np

# --- Custom Anscombe Transform Functions on GPU ---
def anscombe_gpu(image_tensor, constant=3/8):
    """Applies the Anscombe transform for variance stabilization of Poisson data on GPU."""
    return 2.0 * torch.sqrt(image_tensor + constant)

def inverse_anscombe_gpu(transformed_image_tensor, constant=3/8):
    """Applies the inverse Anscombe transform on GPU."""
    return torch.clamp((transformed_image_tensor / 2.0)**2 - constant, min=0.0)

# --- Noise Injection Functions on GPU ---
def add_noise_gpu(image_tensor, noise_type, config, device):
    """
    Applies specified noise to a PyTorch tensor on GPU.
    """
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
        # Estimate scaling factor based on unique values
        vals = len(torch.unique(image_tensor))
        vals = 2 ** np.ceil(np.log2(vals)) if vals > 0 else 256
        noisy = torch.poisson(image_tensor * vals) / vals
        return torch.clamp(noisy, 0.0, 1.0)
        
    elif noise_type == 'mixed_poisson_gaussian':
        g_var = noise_cfg.get('mixed_poisson_gaussian', {}).get('gaussian_var', 0.01)
        # Poisson noise
        vals = len(torch.unique(image_tensor))
        vals = 2 ** np.ceil(np.log2(vals)) if vals > 0 else 256
        poisson_noisy = torch.poisson(image_tensor * vals) / vals
        # Gaussian noise
        std = g_var ** 0.5
        noisy = poisson_noisy + torch.randn_like(poisson_noisy) * std
        return torch.clamp(noisy, 0.0, 1.0)
        
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

# --- GPU Denoising Filter Implementations ---
def median_filter_gpu(img, ksize=5):
    """Applies 2D median filter on GPU."""
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    pad = ksize // 2
    img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')
    patches = img_padded.unfold(2, ksize, 1).unfold(3, ksize, 1)
    patches = patches.contiguous().view(img.size(0), img.size(1), img.size(2), img.size(3), -1)
    median_val, _ = patches.median(dim=-1)
    return median_val.squeeze(0).squeeze(0)

def gaussian_filter_gpu(img, ksize=(5, 5), sigma=1.0):
    """Applies Gaussian filter on GPU."""
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
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
    return denoised.squeeze(0).squeeze(0)

def wiener_filter_gpu(img, mysize=3, noise=None):
    """Applies Wiener filter on GPU."""
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    
    pad = mysize // 2
    img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')
    local_mean = F.avg_pool2d(img_padded, kernel_size=mysize, stride=1)
    local_mean_sq = F.avg_pool2d(img_padded**2, kernel_size=mysize, stride=1)
    local_var = local_mean_sq - local_mean**2
    
    if noise is None:
        noise = torch.mean(local_var)
        
    res = local_mean + (torch.clamp(local_var - noise, min=0.0) / torch.clamp(local_var, min=1e-8)) * (img - local_mean)
    return res.squeeze(0).squeeze(0)

def anscombe_wiener_denoising_gpu(img, balance=0.1):
    """Applies Anscombe, Wiener, and Inverse Anscombe on GPU."""
    transformed_img = anscombe_gpu(img)
    denoised_transformed_img = wiener_filter_gpu(transformed_img, mysize=3, noise=balance)
    denoised_img = inverse_anscombe_gpu(denoised_transformed_img)
    return torch.clamp(denoised_img, 0.0, 1.0)

def bilateral_filter_gpu(img, d=9, sigma_color=75.0, sigma_space=75.0):
    """Applies Bilateral filter on GPU."""
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    H, W = img.shape[2], img.shape[3]
    pad = d // 2
    img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')
    
    x = torch.arange(-pad, pad + 1, dtype=torch.float32, device=img.device)
    y = torch.arange(-pad, pad + 1, dtype=torch.float32, device=img.device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    spatial_w = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma_space**2)).view(1, 1, 1, 1, -1)
    
    patches = img_padded.unfold(2, d, 1).unfold(3, d, 1)
    patches = patches.contiguous().view(1, 1, H, W, -1)
    center = img.unsqueeze(-1)
    
    color_w = torch.exp(-(patches - center)**2 / (2 * (sigma_color / 255.0)**2))
    total_w = spatial_w * color_w
    total_w_sum = total_w.sum(dim=-1, keepdim=True)
    
    denoised = (patches * total_w).sum(dim=-1, keepdim=True) / torch.clamp(total_w_sum, min=1e-8)
    return denoised.squeeze(-1).squeeze(0).squeeze(0)

def non_local_means_gpu(img, h=10, template_window_size=7, search_window_size=21):
    """Applies Non-Local Means filter on GPU."""
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    H, W = img.shape[2], img.shape[3]
    
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
    return denoised.squeeze(0).squeeze(0)

def adaptive_median_filter_gpu(img, s_max=7):
    """Applies Adaptive Median filter on GPU."""
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    
    H, W = img.shape[2], img.shape[3]
    output = img.clone()
    active_mask = torch.ones_like(img, dtype=torch.bool)
    
    for s in range(3, s_max + 1, 2):
        if not active_mask.any():
            break
        pad = s // 2
        img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')
        patches = img_padded.unfold(2, s, 1).unfold(3, s, 1)
        patches = patches.contiguous().view(1, 1, H, W, -1)
        
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
        patches = patches.contiguous().view(1, 1, H, W, -1)
        z_med = patches.median(dim=-1).values
        output[active_mask] = z_med[active_mask]
        
    return output.squeeze(0).squeeze(0)

def kuan_filter_gpu(img, win_size=5, noise_var_estimate=0.04):
    """Applies Kuan filter on GPU."""
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    pad = win_size // 2
    img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')
    local_mean = F.avg_pool2d(img_padded, kernel_size=win_size, stride=1)
    local_sq_mean = F.avg_pool2d(img_padded**2, kernel_size=win_size, stride=1)
    local_variance = local_sq_mean - local_mean**2
    local_variance = torch.clamp(local_variance, min=1e-6)
    
    B = 1.0 - (noise_var_estimate / local_variance)
    B = torch.clamp(B, min=0.0)
    
    denoised = local_mean + B * (img - local_mean)
    return torch.clamp(denoised, 0.0, 1.0).squeeze(0).squeeze(0)

def apply_denoising_gpu(noisy_image_tensor, method, config):
    """
    Applies specified denoising method using config settings on GPU.
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
