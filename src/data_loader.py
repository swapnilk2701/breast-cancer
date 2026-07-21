import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.utils import read_and_preprocess_image

def get_image_paths(raw_dir, supported_formats):
    """
    Scans the raw_dir for class folders and image files matching supported_formats.
    Returns:
        list of dict: [{"class": class_name, "image_name": filename, "path": abs_path}]
    """
    image_items = []
    if not os.path.exists(raw_dir):
        print(f"Error: Raw directory does not exist: {raw_dir}")
        return image_items

    for class_folder in os.listdir(raw_dir):
        class_path = os.path.join(raw_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        for image_name in os.listdir(class_path):
            if not any(image_name.lower().endswith(fmt) for fmt in supported_formats):
                continue
            
            image_items.append({
                "class": class_folder,
                "image_name": image_name,
                "path": os.path.join(class_path, image_name)
            })
            
    return image_items

def load_dataset_generator(image_items, target_size=(256, 256)):
    """
    Generator that yields preprocessed images and metadata for CPU execution.
    """
    for item in image_items:
        img = read_and_preprocess_image(item["path"], target_size)
        if img is None:
            continue
        yield img, item

class MammographyDataset(Dataset):
    """
    PyTorch Dataset for batched loading and preprocessing of mammography images.
    """
    def __init__(self, image_items, target_size=(256, 256)):
        self.image_items = image_items
        self.target_size = target_size

    def __len__(self):
        return len(self.image_items)

    def __getitem__(self, idx):
        item = self.image_items[idx]
        img = read_and_preprocess_image(item["path"], self.target_size)
        if img is None:
            img = np.zeros(self.target_size, dtype=np.float64)
        
        # Convert to float32 PyTorch Tensor with shape (1, H, W)
        tensor_img = torch.from_numpy(img).float().unsqueeze(0)
        return tensor_img, item

def create_gpu_dataloader(image_items, target_size=(256, 256), batch_size=32, num_workers=4):
    """
    Creates an optimized PyTorch DataLoader for GPU pipeline with pinned memory and multi-threading.
    """
    dataset = MammographyDataset(image_items, target_size=target_size)
    
    # Adjust num_workers if running in restricted environments
    actual_workers = min(num_workers, os.cpu_count() or 1)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(actual_workers > 0),
        drop_last=False
    )
    return dataloader
