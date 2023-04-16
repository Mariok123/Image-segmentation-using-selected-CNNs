import sys
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import PredictionDataset
from torch.utils.data import DataLoader
from utils import (
    load_checkpoint,
    get_model,
    save_predictions_as_imgs,
    parse_predict_args,
)

# Prediction hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True

# Entry point of program for using an already trained model for prediction
def main():
    selected_model, checkpoint_path, source_dir_path = parse_predict_args(sys.argv)

    model, *_ = get_model(selected_model)

    # load pretrained model
    load_checkpoint(torch.load(checkpoint_path), model)

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

    pred_dataset = PredictionDataset(source_dir_path, pred_transforms)
    pred_loader = DataLoader(
        pred_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    # save predictions to folder as images
    save_predictions_as_imgs(pred_loader, model, folder="predicted_images/" + selected_model + "/", device=DEVICE)


if __name__ == "__main__":
    main()