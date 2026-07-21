import os
import time
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from src.utils import (
    load_config,
    save_image,
    plot_metrics_comparison,
    display_sample_images
)
from src.data_loader import get_image_paths, create_gpu_dataloader
from src.model_gpu import (
    add_noise_gpu,
    apply_denoising_gpu,
    calculate_metrics_gpu_batch
)

def batch_tensor_to_numpy_list(tensor_batch):
    """
    Converts a batch of GPU PyTorch tensors (B, 1, H, W) to a list of 2D NumPy arrays on CPU.
    """
    cpu_array = tensor_batch.detach().cpu().squeeze(1).numpy()
    cpu_array = np.clip(cpu_array, 0.0, 1.0)
    return [cpu_array[i] for i in range(cpu_array.shape[0])]

def run_pipeline(config_path="config/config.yaml", max_images=None, batch_size=32, num_workers=4):
    """
    Runs the end-to-end mammography denoising pipeline fully accelerated on GPU.
    Optimized for high GPU utilization via batched PyTorch DataLoader, in-VRAM metric calculations,
    and asynchronous multithreaded I/O image saving.
    """
    config = load_config(config_path)
    
    # 1. Enable PyTorch CUDA optimizations
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"Executing High-Performance GPU Mammography Pipeline")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Batch Size: {batch_size} | DataLoader Workers: {num_workers}")
    print(f"=======================================================\n")
    
    # Extract paths
    raw_dir = config['paths']['raw_dir']
    processed_dir = config['paths']['processed_dir']
    results_dir = config['paths']['results_dir']
    
    noisy_out_dir = os.path.join(processed_dir, 'noisy')
    denoised_out_dir = os.path.join(processed_dir, 'denoised')
    
    os.makedirs(noisy_out_dir, exist_ok=True)
    os.makedirs(denoised_out_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    supported_formats = config['data']['supported_formats']
    image_size = tuple(config['data']['image_size'])
    
    noise_types = ['gaussian', 's&p', 'speckle', 'poisson', 'mixed_poisson_gaussian']
    denoising_methods = ['median', 'gaussian', 'wiener', 'bilateral', 'non_local_means', 'anscombe_wiener', 'adaptive_median', 'kuan']
    
    print("Scanning dataset images...")
    image_items = get_image_paths(raw_dir, supported_formats)
    total_images = len(image_items)
    print(f"Found {total_images} images.")
    
    if total_images == 0:
        print("No images found. Pipeline terminating.")
        return

    if max_images is not None:
        image_items = image_items[:max_images]
        print(f"Limiting execution to the first {len(image_items)} images.")
        
    # Create PyTorch DataLoader
    dataloader = create_gpu_dataloader(
        image_items,
        target_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    results_list = []
    total_processed_images = 0
    pipeline_start_time = time.time()
    
    # Initialize ThreadPoolExecutor for background non-blocking disk I/O
    io_pool = ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 1) * 2))
    io_futures = []
    
    # Enable PyTorch inference_mode for maximum speed & minimal VRAM overhead
    with torch.inference_mode():
        pbar = tqdm(dataloader, desc="GPU Processing Batches", unit="batch")
        
        for batch_idx, (batch_tensors, batch_items) in enumerate(pbar):
            batch_start = time.time()
            current_batch_size = batch_tensors.size(0)
            
            # Non-blocking transfer to GPU memory
            original_batch = batch_tensors.to(device, non_blocking=True)
            
            # Extract metadata lists from batch_items
            image_names = batch_items["image_name"]
            classes = batch_items["class"]
            
            for noise_name in noise_types:
                # 1. Batch noise injection on GPU
                noisy_batch = add_noise_gpu(original_batch, noise_name, config, device)
                
                # Asynchronously schedule saving noisy images
                noisy_np_list = batch_tensor_to_numpy_list(noisy_batch)
                noisy_paths = []
                for i in range(current_batch_size):
                    base_name = os.path.splitext(image_names[i])[0]
                    noisy_filename = f"{base_name}_{noise_name}_noisy.png"
                    noisy_path = os.path.join(noisy_out_dir, noisy_filename)
                    noisy_paths.append(noisy_path)
                    io_futures.append(io_pool.submit(save_image, noisy_np_list[i], noisy_path))
                
                for method_name in denoising_methods:
                    # 2. Batch denoising filter execution on GPU
                    denoised_batch = apply_denoising_gpu(noisy_batch, method_name, config)
                    
                    # Asynchronously schedule saving denoised images
                    denoised_np_list = batch_tensor_to_numpy_list(denoised_batch)
                    denoised_paths = []
                    for i in range(current_batch_size):
                        base_name = os.path.splitext(image_names[i])[0]
                        denoised_filename = f"{base_name}_{noise_name}_{method_name}_denoised.png"
                        denoised_path = os.path.join(denoised_out_dir, denoised_filename)
                        denoised_paths.append(denoised_path)
                        io_futures.append(io_pool.submit(save_image, denoised_np_list[i], denoised_path))
                    
                    # 3. Compute metrics entirely in GPU VRAM
                    metrics_batch = calculate_metrics_gpu_batch(original_batch, noisy_batch, denoised_batch)
                    
                    # Record results
                    for i in range(current_batch_size):
                        row = {
                            'Image Name': image_names[i],
                            'Class': classes[i],
                            'Noise Type': noise_name,
                            'Denoising Method': method_name,
                            **metrics_batch[i],
                            'Noisy Image Path': noisy_paths[i],
                            'Denoised Image Path': denoised_paths[i]
                        }
                        results_list.append(row)
                        
            total_processed_images += current_batch_size
            batch_elapsed = time.time() - batch_start
            images_per_sec = (current_batch_size * len(noise_types) * len(denoising_methods)) / max(batch_elapsed, 1e-5)
            
            # Memory logging
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
            pbar.set_postfix({
                "img/s": f"{images_per_sec:.1f}",
                "VRAM": f"{vram_mb:.0f}MB"
            })
            
    # Wait for all background disk I/O tasks to finish
    print("\nFinalizing asynchronous image saving to disk...")
    for future in io_futures:
        future.result()
    io_pool.shutdown()

    total_time = time.time() - pipeline_start_time
    print(f"\nPipeline finished! Processed {total_processed_images} images in {total_time:.2f} seconds.")
    print(f"Overall Throughput: {total_processed_images / total_time:.2f} source images/sec.")
    
    if not results_list:
        print("No results generated.")
        return
        
    # Convert results to DataFrame
    all_results_df = pd.DataFrame(results_list)
    
    # Save raw results
    csv_path = os.path.join(results_dir, 'final_results.csv')
    excel_path = os.path.join(results_dir, 'final_results.xlsx')
    all_results_df.to_csv(csv_path, index=False)
    all_results_df.to_excel(excel_path, index=False)
    print(f"Raw results saved to:\n  - {csv_path}\n  - {excel_path}")
    
    # Compute summary statistics
    summary_metrics = ['PSNR_Denoised_vs_Original', 'SSIM_Denoised_vs_Original', 'MSE_Denoised_vs_Original']
    summary_df = all_results_df.groupby(['Noise Type', 'Denoising Method'])[summary_metrics].mean().reset_index()
    summary_df_sorted = summary_df.sort_values(by=['Noise Type', 'PSNR_Denoised_vs_Original'], ascending=[True, False])
    
    summary_csv_path = os.path.join(results_dir, 'summary_statistics.csv')
    summary_excel_path = os.path.join(results_dir, 'summary_statistics.xlsx')
    summary_df_sorted.to_csv(summary_csv_path, index=False)
    summary_df_sorted.to_excel(summary_excel_path, index=False)
    
    print("\n--- Summary Statistics (Average Metrics) ---")
    print(summary_df_sorted.to_string(index=False))
    print(f"\nSummary statistics saved to:\n  - {summary_csv_path}\n  - {summary_excel_path}")
    
    # Generate performance plots
    print("Generating performance plots...")
    psnr_plot_path = os.path.join(results_dir, 'psnr_comparison.png')
    plot_metrics_comparison(
        all_results_df, 
        'PSNR_Denoised_vs_Original',
        'Average PSNR Comparison (Denoised vs Original)', 
        'Average PSNR (dB)', 
        save_path=psnr_plot_path
    )
    
    ssim_plot_path = os.path.join(results_dir, 'ssim_comparison.png')
    plot_metrics_comparison(
        all_results_df, 
        'SSIM_Denoised_vs_Original',
        'Average SSIM Comparison (Denoised vs Original)', 
        'Average SSIM', 
        save_path=ssim_plot_path
    )
    print(f"Plots saved in: {results_dir}")
    
    # Sample visualization
    sample_row = all_results_df.iloc[0]
    original_img_path = os.path.join(raw_dir, sample_row['Class'], sample_row['Image Name'])
    from src.utils import read_and_preprocess_image
    orig = read_and_preprocess_image(original_img_path, image_size)
    noisy = read_and_preprocess_image(sample_row['Noisy Image Path'], image_size)
    denoised = read_and_preprocess_image(sample_row['Denoised Image Path'], image_size)
    
    if orig is not None and noisy is not None and denoised is not None:
        sample_viz_path = os.path.join(results_dir, 'sample_denoising_result.png')
        display_sample_images(
            orig, noisy, denoised, 
            title_suffix=f"({sample_row['Noise Type']} / {sample_row['Denoising Method']})", 
            save_path=sample_viz_path
        )
        print(f"Sample visualization saved to: {sample_viz_path}")

if __name__ == "__main__":
    run_pipeline(max_images=32, batch_size=32)
