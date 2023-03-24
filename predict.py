import sys
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import PredictionDataset
from unet_model import UNET
from doubleunet_model import DoubleUNET
from resunetpp_model import ResUNETpp
from torch.utils.data import DataLoader
from utils import (
    load_checkpoint,
    save_predictions_as_imgs,
    parse_args,
)

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True

def main():
    selected_model, _, load_model_path, _, source_dir = parse_args(sys.argv)

    if selected_model == "UNET":
        model = UNET().to(DEVICE)
    elif selected_model == "DoubleUNET":
        model = DoubleUNET().to(DEVICE)
    elif selected_model == "ResUNETpp":
        model = ResUNETpp().to(DEVICE)

    load_checkpoint(torch.load(load_model_path), model)

    pred_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    pred_dataset = PredictionDataset(source_dir, pred_transforms)
    pred_loader = DataLoader(
        pred_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    # print predictions to folder as images
    save_predictions_as_imgs(
        pred_loader, model, folder="predicted_images/" + selected_model + "/", device=DEVICE
    )


if __name__ == "__main__":
    main()