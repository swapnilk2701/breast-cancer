import os
import pandas as pd
from src.utils import (
    load_config,
    save_image,
    calculate_metrics,
    plot_metrics_comparison,
    display_sample_images
)
from src.data_loader import get_image_paths, load_dataset_generator
from src.model import add_noise, apply_denoising

def run_pipeline(config_path="config/config.yaml", max_images=None):
    """
    Runs the end-to-end mammography denoising pipeline.
    """
    config = load_config(config_path)
    
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
    
    # Noise and denoising types to evaluate
    noise_types = ['gaussian', 's&p', 'speckle', 'poisson', 'mixed_poisson_gaussian']
    denoising_methods = ['median', 'gaussian', 'wiener', 'bilateral', 'non_local_means', 'anscombe_wiener', 'adaptive_median']
    
    print("Scanning dataset images...")
    image_items = get_image_paths(raw_dir, supported_formats)
    total_images = len(image_items)
    print(f"Found {total_images} images.")
    
    if max_images is not None:
        image_items = image_items[:max_images]
        print(f"Limiting execution to the first {len(image_items)} images.")
        
    results_list = []
    
    dataset_gen = load_dataset_generator(image_items, image_size)
    
    for idx, (original_img, item) in enumerate(dataset_gen):
        class_folder = item["class"]
        image_name = item["image_name"]
        base_name = os.path.splitext(image_name)[0]
        
        print(f"[{idx+1}/{len(image_items)}] Processing {image_name} (Class: {class_folder})")
        
        for noise_name in noise_types:
            # Inject noise
            noisy_img = add_noise(original_img, noise_name, config)
            
            # Save noisy image
            noisy_filename = f"{base_name}_{noise_name}_noisy.png"
            noisy_path = os.path.join(noisy_out_dir, noisy_filename)
            save_image(noisy_img, noisy_path)
            
            for method_name in denoising_methods:
                # Apply denoising filter
                denoised_img = apply_denoising(noisy_img, method_name, config)
                
                # Save denoised image
                denoised_filename = f"{base_name}_{noise_name}_{method_name}_denoised.png"
                denoised_path = os.path.join(denoised_out_dir, denoised_filename)
                save_image(denoised_img, denoised_path)
                
                # Compute metrics
                metrics = calculate_metrics(original_img, noisy_img, denoised_img)
                
                # Record result entry
                row = {
                    'Image Name': image_name,
                    'Class': class_folder,
                    'Noise Type': noise_name,
                    'Denoising Method': method_name,
                    **metrics,
                    'Noisy Image Path': noisy_path,
                    'Denoised Image Path': denoised_path
                }
                results_list.append(row)
                
    if not results_list:
        print("No images were successfully processed. Pipeline terminating.")
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
    
    # Generate and save comparison plots
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
    
    # Generate sample visualization for verification
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
    # Run on all images in the dataset
    run_pipeline(max_images=None)
