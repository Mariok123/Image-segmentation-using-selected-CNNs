import sys
import datetime 
import torch
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from early_stopping import EarlyStopping
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_model,
    get_loaders,
    check_training_metrics,
    save_training_metrics,
    save_val_ds_as_imgs,
    save_val_predictions_as_imgs,
    parse_args,
)


# Training hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
EARLY_STOP_PATIENCE = 3

# Does one epoch of training
# returns how long training the epoch took and it's loss
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward pass
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    epoch_loss = float('{0:.4f}'.format(total_loss/len(loader)))
    return loop.format_interval(loop.format_dict['elapsed']), epoch_loss

# Entry point of program for training the neural networks
def main():
    selected_model, selected_dataset, load_model, num_epochs, early_stop, _  = parse_args(sys.argv)

    currTime = datetime.datetime.now()
    currTime = currTime.strftime('%Y%m%dT%H%M%S')
    training_results_folder = "training_results/" + selected_model + "/" + selected_dataset + "/" + currTime + "/"

    # image augmentations for the training dataset
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

    # image augmentations for the validation dataset
    val_transform = A.Compose(
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

    model, loss_fn, optimizer = get_model(selected_model, DEVICE, LEARNING_RATE)
    train_loader, val_loader = get_loaders(selected_dataset, BATCH_SIZE, train_transform, val_transform, NUM_WORKERS, PIN_MEMORY)

    # load an already trained model for further training
    if load_model:
        load_checkpoint(torch.load(selected_model), model)
        check_training_metrics(val_loader, model, device=DEVICE)
    
    scaler = torch.cuda.amp.GradScaler()

    # save original images and masks from validation dataset for reference
    save_val_ds_as_imgs(val_loader, folder=training_results_folder + "validation_set/", device=DEVICE)

    if early_stop:
        early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    for epoch in range(num_epochs):
        training_metrics = []

        print("=======================")
        print(f"Training epoch {epoch}/{num_epochs-1}")

        # train an epoch
        epoch_time, epoch_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        training_metrics.extend((epoch_time, epoch_loss))

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, selected_model + "_"+ selected_dataset + ".pth.tar")

        # check training results
        val_loss, accuracy, f1, iou = check_training_metrics(val_loader, model, loss_fn, device=DEVICE)
        training_metrics.extend((val_loss, accuracy, f1, iou))

        # save predicted results of validation dataset to a folder
        save_val_predictions_as_imgs(val_loader, model, folder=training_results_folder + str(epoch) + "/", device=DEVICE)

        # save training metrics
        save_training_metrics(training_results_folder + "training_metrics.csv", training_metrics)

        if early_stop:
            # test if validation loss got better
            if early_stopper(model, val_loss):
                print(f"Stopping early due to not improving for {early_stopper.counter} epochs")
                print(f"Resaving previous best model")

                # resave best model
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, selected_model + "_"+ selected_dataset + ".pth.tar")
                
                break

if __name__ == "__main__":
    main()