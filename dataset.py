import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from PIL import Image
import pandas as pd
import os

# -----------------------
# Transforms
# -----------------------
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# -----------------------
# Safe loader for ImageFolder
# -----------------------
def safe_loader(path):
    try:
        return default_loader(path)
    except Exception as e:
        print(f"[Warning] Skipping unreadable image: {path} ({e})")
        return None

class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = safe_loader(path)
        if sample is None:
            # Skip unreadable image
            return self.__getitem__((index + 1) % len(self.samples))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

# -----------------------
# Split dataset into train/val/test
# -----------------------
def get_datasets(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    dataset = SafeImageFolder(data_dir, transform=get_transforms())
    total = len(dataset)
    train_len = int(total * train_ratio)
    val_len = int(total * val_ratio)
    test_len = total - train_len - val_len
    train_set, val_set, test_set = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed)
    )
    return train_set, val_set, test_set

# -----------------------
# Dataset for CSV-based test set
# -----------------------
class RVLCDIPDataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Pad doc_id to match filenames (10-digit filenames)
        filename = f"{int(row['doc_id']):010d}.tif"
        img_path = os.path.join(self.img_root, filename)

        # Skip missing images
        if not os.path.exists(img_path):
            print(f"[Warning] File not found: {img_path}. Skipping...")
            return None

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[Warning] Cannot read image {img_path}: {e}")
            return None

        label = row['label']

        if self.transform:
            image = self.transform(image)

        return image, label, row['doc_id']

# -----------------------
# Collate function for DataLoader to skip None
# -----------------------
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
