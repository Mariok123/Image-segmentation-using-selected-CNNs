import sys
import datetime 
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from early_stopping import EarlyStopping
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_model,
    get_loaders,
    validate_model,
    save_training_metrics,
    save_val_ds_as_imgs,
    parse_training_args,
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
DATASET_SPLIT = 0.8

# Does one epoch of training
# returns how long training the epoch took and it's loss
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward pass
        #with torch.cuda.amp.autocast(): # can cause NaN value in loss
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

    epoch_time = float('{0:.2f}'.format(loop.format_dict['elapsed']))
    epoch_loss = float('{0:.4f}'.format(total_loss/len(loader)))
    return epoch_time, epoch_loss

# Entry point of program for training the neural networks
def main():
    selected_model, selected_dataset, dataset_source_dir, dataset_mask_dir, num_epochs, checkpoint_path, early_stop, save_validation_results = parse_training_args(sys.argv)

    startTimestamp = datetime.datetime.now()
    startTimestamp = startTimestamp.strftime('%Y%m%dT%H%M%S')
    training_results_folder = "training_results/" + selected_model + "/" + startTimestamp + "/"

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

    scaler = torch.cuda.amp.GradScaler()

    model, loss_fn, optimizer = get_model(selected_model, DEVICE, LEARNING_RATE)
    train_loader, val_loader = get_loaders(selected_dataset, BATCH_SIZE, train_transform, val_transform, NUM_WORKERS, PIN_MEMORY, dataset_split=DATASET_SPLIT, ds_source_dir=dataset_source_dir, ds_mask_dir=dataset_mask_dir)

    # load an already trained model for further training
    if checkpoint_path:
        load_checkpoint(torch.load(checkpoint_path), model)
        validate_model(val_loader, model, loss_fn, device=DEVICE)

    # save original images and masks from validation dataset for reference
    if save_validation_results:
        save_val_ds_as_imgs(val_loader, folder=training_results_folder + "validation_set/", device=DEVICE)

    if early_stop:
        early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    # prepare .csv file for saving training metrics
    save_training_metrics(training_results_folder, ["epoch_time", "epoch_loss", "val_loss", "accuracy", "f1_score", "iou_score"])

    # start the training
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
        save_checkpoint(checkpoint, selected_model + "_" + startTimestamp + ".pth.tar")

        # validate training results
        val_loss, accuracy, f1, iou = validate_model(val_loader, model, loss_fn, save_validation_results, folder=training_results_folder + str(epoch) + "/", device=DEVICE)
        training_metrics.extend((val_loss, accuracy, f1, iou))

        # save training metrics
        save_training_metrics(training_results_folder, training_metrics)

        if early_stop:
            # test for validation loss improvement
            if early_stopper(model, val_loss):
                print(f"Stopping early due to not improving for {EARLY_STOP_PATIENCE} epochs")
                print(f"Resaving previous best model")

                # resave best model
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, selected_model + "_" + startTimestamp + ".pth.tar")
                break

if __name__ == "__main__":
    main()