import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class InitDataset(Dataset):
    def __init__(self, image_dir):
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]

class SubDataset(Dataset):
    def __init__(self, image_dir, mask_dir, subset, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.subsetImages = subset
        self.transform = transform

    def __len__(self):
        return len(self.subsetImages)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.subsetImages[index])
        mask_path = os.path.join(self.mask_dir, self.subsetImages[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # L - Greyscale
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

class PredictionDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image