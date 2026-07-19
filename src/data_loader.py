import os
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
    Generator that yields preprocessed images and metadata.
    """
    for item in image_items:
        img = read_and_preprocess_image(item["path"], target_size)
        if img is not None:
            yield img, item
