import os
import pandas as pd
import numpy as np
from src.utils import (
    load_config,
    save_image,
    calculate_metrics,
    plot_metrics_comparison,
    display_sample_images,
    display_sample_mse_images,
    display_enhancement_comparison,
    read_and_preprocess_image
)
from src.data_loader import get_image_paths, load_dataset_generator
from src.model import add_noise, apply_denoising, apply_contrast_enhancement
from src.pectoral_removal import PectoralMuscleRemover
from src.intensity_normalization import IntensityNormalizer
from src.contrast_sharpening.py import MammogramEnhancer

def run_pipeline(config_path="config/config.yaml", max_images=None):
    """
    Runs the end-to-end mammography preprocessing, denoising, and contrast enhancement pipeline.
    Preserves all noise and denoising stages while integrating Stage 5 contrast enhancement.
    """
    config = load_config(config_path)
    
    # Extract paths
    raw_dir = config['paths']['raw_dir']
    processed_dir = config['paths']['processed_dir']
    results_dir = config['paths']['results_dir']
    
    noisy_out_dir = os.path.join(processed_dir, 'noisy')
    denoised_out_dir = os.path.join(processed_dir, 'denoised')
    enhanced_out_dir = os.path.join(processed_dir, 'enhanced')
    pectoral_out_dir = os.path.join(processed_dir, 'pectoral_removed')
    normalized_out_dir = os.path.join(processed_dir, 'normalized')
    
    os.makedirs(noisy_out_dir, exist_ok=True)
    os.makedirs(denoised_out_dir, exist_ok=True)
    os.makedirs(enhanced_out_dir, exist_ok=True)
    os.makedirs(pectoral_out_dir, exist_ok=True)
    os.makedirs(normalized_out_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    supported_formats = config['data']['supported_formats']
    image_size = tuple(config['data']['image_size'])
    
    # Preprocessing initializers
    pec_cfg = config.get('pectoral_removal', {})
    pec_enabled = pec_cfg.get('enabled', True)
    pec_remover = PectoralMuscleRemover(
        roi_fraction_h=pec_cfg.get('roi_fraction_h', 0.5),
        roi_fraction_w=pec_cfg.get('roi_fraction_w', 0.5),
        fill_value=pec_cfg.get('fill_value', 0)
    ) if pec_enabled else None

    norm_cfg = config.get('intensity_normalization', {})
    norm_enabled = norm_cfg.get('enabled', True)
    normalizer = IntensityNormalizer(
        method=norm_cfg.get('method', 'robust_min_max'),
        p_low=norm_cfg.get('p_low', 1.0),
        p_high=norm_cfg.get('p_high', 99.0),
        target_min=norm_cfg.get('target_min', 0.0),
        target_max=norm_cfg.get('target_max', 255.0)
    ) if norm_enabled else None

    enh_cfg = config.get('enhancement', {})
    enh_enabled = enh_cfg.get('enabled', True)
    enh_method = enh_cfg.get('method', 'clahe_unsharp_mask')

    # Noise and denoising types to evaluate
    noise_types = ['gaussian', 's&p', 'speckle', 'poisson', 'mixed_poisson_gaussian']
    denoising_methods = ['median', 'gaussian', 'wiener', 'bilateral', 'non_local_means', 'anscombe_wiener', 'adaptive_median', 'kuan']
    
    print("Scanning dataset images...")
    image_items = get_image_paths(raw_dir, supported_formats)
    total_images = len(image_items)
    print(f"Found {total_images} images.")
    
    if max_images is not None:
        image_items = image_items[:max_images]
        print(f"Limiting execution to the first {len(image_items)} images.")
        
    results_list = []
    
    dataset_gen = load_dataset_generator(image_items, image_size)
    
    for idx, (raw_img, item) in enumerate(dataset_gen):
        class_folder = item["class"]
        image_name = item["image_name"]
        base_name = os.path.splitext(image_name)[0]
        
        print(f"[{idx+1}/{len(image_items)}] Processing {image_name} (Class: {class_folder})")
        
        # Preprocessing Stage 1: Pectoral Muscle Removal
        current_img = raw_img
        if pec_enabled and pec_remover is not None:
            raw_u8 = (np.clip(raw_img, 0, 1) * 255).astype(np.uint8)
            pec_u8, _ = pec_remover.process_image(raw_u8)
            current_img = pec_u8.astype(np.float64) / 255.0
            pec_path = os.path.join(pectoral_out_dir, f"{base_name}_pectoral_removed.png")
            save_image(current_img, pec_path)

        # Preprocessing Stage 2: Intensity Normalization
        if norm_enabled and normalizer is not None:
            cur_u8 = (np.clip(current_img, 0, 1) * 255).astype(np.uint8)
            norm_u8 = normalizer.normalize_image(cur_u8)
            current_img = norm_u8.astype(np.float64) / 255.0
            norm_path = os.path.join(normalized_out_dir, f"{base_name}_normalized.png")
            save_image(current_img, norm_path)

        original_img = current_img

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
                
                # Apply Stage 5 Contrast Enhancement & Sharpening
                enhanced_img = None
                enhanced_path = None
                if enh_enabled:
                    enhanced_img = apply_contrast_enhancement(denoised_img, enh_method, config)
                    enhanced_filename = f"{base_name}_{noise_name}_{method_name}_enhanced.png"
                    enhanced_path = os.path.join(enhanced_out_dir, enhanced_filename)
                    save_image(enhanced_img, enhanced_path)

                # Compute metrics (includes noise, denoising, and enhancement metrics)
                metrics = calculate_metrics(original_img, noisy_img, denoised_img, enhanced_img=enhanced_img)
                
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
                if enhanced_path:
                    row['Enhanced Image Path'] = enhanced_path

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
    if 'PSNR_Enhanced_vs_Original' in all_results_df.columns:
        summary_metrics.extend([
            'PSNR_Enhanced_vs_Original', 'SSIM_Enhanced_vs_Original', 'MSE_Enhanced_vs_Original',
            'Enhanced_Entropy', 'Entropy_Delta', 'CII'
        ])

    available_summary_metrics = [m for m in summary_metrics if m in all_results_df.columns]
    summary_df = all_results_df.groupby(['Noise Type', 'Denoising Method'])[available_summary_metrics].mean().reset_index()
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

    mse_plot_path = os.path.join(results_dir, 'mean_squared_comparison.png')
    plot_metrics_comparison(
        all_results_df, 
        'MSE_Denoised_vs_Original',
        'Average Mean Squared Error (MSE) Comparison (Denoised vs Original)', 
        'Average MSE', 
        save_path=mse_plot_path
    )

    # Also save as mse_comparison.png for alternative naming reference
    plot_metrics_comparison(
        all_results_df, 
        'MSE_Denoised_vs_Original',
        'Average MSE Comparison (Denoised vs Original)', 
        'Average MSE', 
        save_path=os.path.join(results_dir, 'mse_comparison.png')
    )

    if 'CII' in all_results_df.columns:
        cii_plot_path = os.path.join(results_dir, 'cii_comparison.png')
        plot_metrics_comparison(
            all_results_df,
            'CII',
            'Average Contrast Improvement Index (CII) Comparison',
            'Average CII',
            save_path=cii_plot_path
        )

    print(f"Plots saved in: {results_dir}")
    
    # Generate sample visualization for verification
    sample_row = all_results_df.iloc[0]
    original_img_path = os.path.join(raw_dir, sample_row['Class'], sample_row['Image Name'])
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

        sample_mse_viz_path = os.path.join(results_dir, 'sample_mean_squared_image.png')
        display_sample_mse_images(
            orig, noisy, denoised,
            title_suffix=f"({sample_row['Noise Type']} / {sample_row['Denoising Method']})",
            save_path=sample_mse_viz_path
        )
        print(f"Sample MSE visualization saved to: {sample_mse_viz_path}")

        # 5-panel enhancement comparison
        enhancer = MammogramEnhancer()
        denoised_u8 = (np.clip(denoised, 0, 1) * 255).astype(np.uint8)
        enh_methods, _ = enhancer.compare_all_methods(denoised_u8)
        enh_5panel_path = os.path.join(results_dir, 'sample_enhancement_comparison.png')
        display_enhancement_comparison(
            denoised_u8,
            enh_methods['HE'],
            enh_methods['CLAHE'],
            enh_methods['Unsharp_Mask'],
            enh_methods['CLAHE_plus_UM'],
            title_suffix=f"({sample_row['Noise Type']} / {sample_row['Denoising Method']})",
            save_path=enh_5panel_path
        )
        print(f"Sample 5-panel enhancement visualization saved to: {enh_5panel_path}")

if __name__ == "__main__":
    # Run on images in the dataset
    run_pipeline(max_images=10)
