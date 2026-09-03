# Breast Cancer Mammography Denoising AI Project.

This repository contains an end-to-end modular pipeline designed to evaluate and compare the performance of image denoising techniques on mammography datasets under various synthetic noise conditions. The pipeline supports both **CPU** execution (via NumPy/OpenCV/SciPy) and **GPU** acceleration (via PyTorch CUDA operations).

---

## Processing Workflow & Architecture

The processing pipeline executes the following sequential steps for each input mammogram:

1. **Dataset Scanning & Preprocessing**: Loads grayscale mammogram images (e.g., `benign`, `malignant`) and resizes them to standard dimensions.
2. **Noise Injection**: Synthetically corrupts images using 5 distinct noise models:
   - **Gaussian Noise**: Additive electronic/thermal sensor noise.
   - **Salt & Pepper (S&P) Noise**: Impulse noise from digital signal transmission errors.
   - **Speckle Noise**: Multiplicative noise characteristic of ultrasound / radar imaging.
   - **Poisson Noise**: Quantum photon shot noise inherent in low-dose X-ray mammography.
   - **Mixed Poisson-Gaussian Noise**: Realistic compound noise simulating X-ray photon counting + electronic sensor noise.
3. **Denoising Filter Execution**: Applies 8 spatial and frequency-domain filtering algorithms:
   - **Median Filter**: Non-linear impulse noise reduction.
   - **Gaussian Filter**: Linear spatial smoothing filter.
   - **Wiener Filter**: Minimum mean square error frequency-domain adaptive filter.
   - **Bilateral Filter**: Edge-preserving spatial and range smoothing filter.
   - **Non-Local Means (NLM)**: Non-local self-similarity patch matching filter.
   - **Anscombe-Wiener Filter**: Variance-stabilizing Anscombe transformation followed by Wiener filtering and inverse Anscombe mapping.
   - **Adaptive Median Filter**: Dynamic window-expanding median filter for impulse noise.
   - **Kuan Filter**: Local statistics-based adaptive speckle reduction filter.
4. **Quantitative Metric Evaluation**: Computes image quality metrics for every noise-filter pair:
   - **PSNR (Peak Signal-to-Noise Ratio)**: Measures signal fidelity (higher is better).
   - **SSIM (Structural Similarity Index)**: Measures structural/perceptual similarity to the original image (range [0, 1], higher is better).
   - **MSE (Mean Squared Error)**: Quantifies pixel intensity differences (lower is better).
5. **Results Generation**: Exports raw metric CSV/Excel files, summary statistics, performance comparison plots (`psnr_comparison.png`, `ssim_comparison.png`), and side-by-side sample result visualizations.

---

## Project Structure

```
breast-cancer-mammography-denoising/
├── Dockerfile              # Default Dockerfile for CPU execution
├── Dockerfile.cpu          # CPU-specific container configuration
├── Dockerfile.gpu          # PyTorch CUDA GPU container configuration
├── README.md               # Documentation and setup guide
├── requirements.txt        # CPU Python dependencies
├── requirements-gpu.txt    # GPU Python dependencies
├── config/
│   └── config.yaml         # Configuration file for dataset paths, noise params, and filter settings
├── data/
│   ├── raw/                # Input raw mammography images (e.g., benign/, malignant/)
│   └── processed/          # Pipeline outputs
│       ├── noisy/          # Generated noise-corrupted images
│       ├── denoised/       # Filtered output images
│       └── results/        # CSV statistics, Excel summaries, and performance plots
├── src/
│   ├── __init__.py         # Package initializer
│   ├── data_loader.py      # Dataset scanner and image loading generator
│   ├── model.py            # CPU-based noise injection and denoising filter implementations
│   ├── model_gpu.py        # PyTorch GPU-accelerated noise and denoising filters
│   ├── utils.py            # Image I/O, PSNR/SSIM/MSE metric calculators, and plot helpers
│   ├── engine.py           # Main CPU execution pipeline
│   └── engine_gpu.py       # Main GPU-accelerated execution pipeline
└── tests/
    └── test_model.py       # Unit tests verifying filter and noise logic
```

---

## Setup & Local Execution

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA driver (Optional, for GPU acceleration)

---

### Option A: CPU Execution (Local)

1. **Install CPU Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Pipeline on CPU**:
   ```bash
   python -m src.engine
   ```

---

### Option B: GPU Execution (Local)

1. **Install PyTorch with CUDA Support**:
   ```bash
   pip install -r requirements-gpu.txt
   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124
   ```
2. **Run Pipeline on GPU**:
   ```bash
   python -m src.engine_gpu
   ```

---

## Docker Container Setup

You can run the pipeline inside isolated Docker containers using either CPU or GPU acceleration.

### 1. CPU Docker Container

**Build CPU Image:**
```bash
docker build -f Dockerfile.cpu -t breast-cancer-denoising:cpu .
```

**Run CPU Container:**
* **Windows (CMD):**
  ```cmd
  docker run --rm -v "%cd%\data:/app/data" breast-cancer-denoising:cpu
  ```
* **Linux / macOS (Bash):**
  ```bash
  docker run --rm -v "$(pwd)/data:/app/data" breast-cancer-denoising:cpu
  ```

---

### 2. GPU Docker Container (NVIDIA CUDA)

> **Note:** Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host system.

**Build GPU Image:**
```bash
docker build -f Dockerfile.gpu -t breast-cancer-denoising:gpu .
```

**Run GPU Container:**
* **Windows (CMD):**
  ```cmd
  docker run --rm --gpus all --shm-size=8g -v "%cd%\data:/app/data" breast-cancer-denoising:gpu
  ```
* **Linux / macOS (Bash):**
  ```bash
  docker run --rm --gpus all --shm-size=8g -v "$(pwd)/data:/app/data" breast-cancer-denoising:gpu
  
  
  ```
  ```bash
  docker run --rm --gpus all --shm-size=8g -v "$(pwd)/data:/app/data" breast-cancer-denoising:gpu python -m src.section3_contrast_sharpening
  ```

---

## Running Unit Tests

To verify filter math and noise function behavior:
```bash
python -m unittest tests/test_model.py
```
