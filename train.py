import sys
import datetime 
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet_model import UNET
from doubleunet_model import DoubleUNET
from resunetpp_model import ResUNETpp
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_carvana_loaders,
    get_imcdb_loaders,
    check_accuracy,
    save_val_predictions_as_imgs,
    parse_args,
)
from modules import (
    DiceLoss,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True

CARVANA_DIR = [
    "data/Carvana/train_images/",
    "data/Carvana/train_masks/"
]

IMCDB_DIR = "data/IMCDB-main"

# Does one epoch of training
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    selected_model, selected_dataset, load_model, NUM_EPOCHS, _ = parse_args(sys.argv)
    currTime = datetime.datetime.now()
    currTime = currTime.strftime('%Y%m%dT%H%M%S')

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            #A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
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

    if selected_model == "UNET":
        model = UNET().to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif selected_model == "DoubleUNET":
        model = DoubleUNET().to(DEVICE)
        #loss_fn = DiceLoss()
        #optimizer = optim.Adam(model.parameters(), lr=1e-5)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.NAdam(model.parameters(), lr=LEARNING_RATE)
    elif selected_model == "ResUNETpp":
        model = ResUNETpp().to(DEVICE)
        loss_fn = DiceLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    else:
        print("UNKNOWN MODEL")
        exit(1)

    if selected_dataset == "Carvana":
        train_loader, val_loader = get_carvana_loaders(
            CARVANA_DIR[0],
            CARVANA_DIR[1],
            BATCH_SIZE,
            train_transform,
            val_transforms,
            NUM_WORKERS,
            PIN_MEMORY
        ) 
    elif selected_dataset == "IMCDB":
        train_loader, val_loader = get_imcdb_loaders(
            IMCDB_DIR,
            BATCH_SIZE,
            train_transform,
            val_transforms,
            NUM_WORKERS,
            PIN_MEMORY
        )
    else:
        print("UNKNOWN DATASET")
        exit(1)

    if load_model:
        load_checkpoint(torch.load(selected_model), model)
        check_accuracy(val_loader, model, device=DEVICE)
    
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print("=======================")
        print(f"Training epoch {epoch}.")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, selected_model + "_"+ selected_dataset + ".pth.tar")

        # check accuracy
        acc = check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_val_predictions_as_imgs(
            val_loader, model, folder="saved_images/" + selected_model + "/" + selected_dataset + "/" + currTime + "/" + str(epoch) + " [" + str(acc) + "]/", device=DEVICE
        )


if __name__ == "__main__":
    main()