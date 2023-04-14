import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# Dataset class for loading the Carvana dataset
# can be used for any dataset that has the training images and masks in seperated folder with same ordered content
# (ie. the first mask corresponds to the first image, second to second, etc.)
class CarvanaDataset(Dataset):
    def __init__(self, image_dir, masks_dir):
        self.images = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
        self.masks = [os.path.join(masks_dir, file) for file in os.listdir(masks_dir)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.masks[index]
    
# Dataset class for loading the IMCDB dataset
# the way the directories are ordered in the dataset is atrociously unique, thus it cannot really be reused
class IMCDBDataset(Dataset):
    def __init__(self, data_dir):
        self.images = []
        self.masks = []

        dir_list = [os.path.join(data_dir, directory) for directory in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, directory))]
        for directory_path in dir_list:
            subdir_path_list = [os.path.join(directory_path, subdirectory) for subdirectory in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, subdirectory))]
            subdir_path_list.sort(key = str.lower)

            for subdirectory in subdir_path_list:
                mask_folder_found = ''
                if "Page" in subdirectory:
                    for subdir in subdir_path_list:
                        if "mask" in subdir:
                            mask_folder_found = subdir
                            break
                        elif "MASK" in subdir:
                            mask_folder_found = subdir

                elif "PAGE" in subdirectory:
                    for subdir in subdir_path_list:
                        if "MASK" in subdir:
                            mask_folder_found = subdir
                            break
                        elif "mask" in subdir:
                            mask_folder_found = subdir

                if mask_folder_found:
                    subdir_path_list.remove(mask_folder_found)
                    temp_images = [os.path.join(subdirectory, file) for file in os.listdir(subdirectory) if os.path.isfile(os.path.join(subdirectory, file))]
                    temp_masks = [os.path.join(mask_folder_found, file) for file in os.listdir(mask_folder_found) if os.path.isfile(os.path.join(mask_folder_found, file))]

                    amount_to_use = len(temp_images) if len(temp_images) <= len(temp_masks) else len(temp_masks)

                    self.images.extend(temp_images[:amount_to_use])
                    self.masks.extend(temp_masks[:amount_to_use])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.masks[index]

# Dataset class for seperated training or validation dataset
# requires for the images and masks to be already loaded into a single 2d list
# [i][0] corresponds to the image path and [i][0] corresponds to the mask path
class SubDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image = np.array(Image.open(self.subset[index][0]).convert("RGB"))
        
        # L - Greyscale
        mask = np.array(Image.open(self.subset[index][1]).convert("L"), dtype=np.float32)
        # convert the mask to binary map
        mask = np.where(mask == 255, 1.0, 0.0)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

# Dataset class for storing the images to be segmented
# requires for the images to be in a single folder
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

        return image, self.images[index]