# Breast Cancer Mammography Denoising AI Project

This repository contains a modular Python-based pipeline designed to evaluate and compare the effectiveness of different image denoising techniques on mammography datasets under various noise conditions.

## Project Structure

Following the structural layout defined in `project-structure.md`:

```
breast-cancer-mammography-denoising/
├── .env                  # Local environment configuration settings
├── .gitignore            # Excludes data/processed, results, and local env files
├── Dockerfile            # Containerizes the application for deployment
├── README.md             # Setup guide, usage instructions, and documentation
├── requirements.txt      # List of dependencies
├── config/
│   └── config.yaml       # Hyperparameters, paths, and environment settings
├── data/
│   └── raw/              # Immutable original source datasets (e.g. benign, malignant folders)
│   └── processed/        # Generated noisy and denoised images
│       ├── noisy/        # Corrupted images by noise type
│       ├── denoised/      # Denoised results from different filters
│       └── results/      # Comparative metrics tables, summary stats, and plots
├── notebooks/
│   └── breast_cancer.py  # Original Google Colab notebook script
├── src/
│   ├── __init__.py       # Package initializer
│   ├── data_loader.py    # Fetches and batches training datasets
│   ├── model.py          # Noise injection and denoising filter implementations
│   ├── utils.py          # Helper functions (loading config, saving images, calculating metrics, plotting)
│   └── engine.py         # Main pipeline execution script
└── tests/
    └── test_model.py     # Unit tests verifying noise injection and filter integrity
```

## Getting Started

### Prerequisites

You need Python 3.8+ installed.

### Setup

1. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure your source raw mammogram images are inside `data/raw/benign/` and `data/raw/malignant/` directories.

### Running Unit Tests

To run the unit tests and verify the filter functions and noise logic:
```bash
python -m unittest tests/test_model.py
```

### Running the Denoising Pipeline

To run the full end-to-end pipeline:
```bash
python src/engine.py
```

This script will:
- Read images from `data/raw`
- Corrupt them using 5 noise types: Gaussian, Salt & Pepper, Speckle, Poisson, and Mixed Poisson-Gaussian
- Apply 7 denoising filters: Median, Gaussian, Wiener, Bilateral, Fast Non-Local Means, Anscombe-Wiener, and Adaptive Median
- Save processed outputs into `data/processed/`
- Compute quantitative evaluation metrics (MSE, PSNR, SSIM)
- Export detailed results tables (`final_results.csv`, `summary_statistics.xlsx`) and comparison bar charts (`psnr_comparison.png`, `ssim_comparison.png`).

## Docker Containerization

To build and run the docker image:
```bash
docker build -t breast-cancer-mammography-denoising .
docker run --rm -v ${PWD}/data:/app/data breast-cancer-mammography-denoising
```
