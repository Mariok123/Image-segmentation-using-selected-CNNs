import torch
import torchvision
import os
import csv
from dataset import CarvanaDataset, SubDataset, IMCDBDataset
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, JaccardIndex, F1Score
import torch.nn as nn
import torch.optim as optim
from modules import DiceLoss
from unet_model import UNET
from doubleunet_model import DoubleUNET
from resunetpp_model import ResUNETpp

# Constant for storing the model names that are available to use
VALID_MODELS = [
    "UNET",
    "DoubleUNET",
    "ResUNETpp"
]

# Constant for storing the dataset names that are available to use
VALID_DATASETS = [
    "Carvana",
    "IMCDB"
]

CARVANA_DIR = [
    "data/Carvana/train_images/",
    "data/Carvana/train_masks/"
]

IMCDB_DIR = "data/IMCDB-main"

# Saves the checkpoint of the currently training model
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)

# Loads the checkpoint of a previously trained model
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


# Returns necessary parameters that are unique between the different implemented models
# if another model was to be implemented, it has to be also added here
def get_model(selected_model, device="cuda", learning_rate = 1e-4):
    if selected_model == "UNET":
        model = UNET().to(device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif selected_model == "DoubleUNET":
        model = DoubleUNET().to(device)

        #loss_fn = DiceLoss()
        #optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
    elif selected_model == "ResUNETpp":
        model = ResUNETpp().to(device)
        loss_fn = DiceLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return model, loss_fn, optimizer

# Returns the training and validation dataloader for one of the supported datasets
# if another dataset was to be added, it has to be also added here
def get_loaders(selected_dataset, batch_size, train_transform, val_transform, num_workers, pin_memory, dataset_split=0.8):
    if selected_dataset == "Carvana":
        init_ds = CarvanaDataset(CARVANA_DIR[0], CARVANA_DIR[1])
    elif selected_dataset == "IMCDB":
        init_ds = IMCDBDataset(IMCDB_DIR)
    
    # split dataset into training and validation subsets
    train_split = int(len(init_ds)*dataset_split)
    train_subset, val_subset = random_split(init_ds, [train_split, len(init_ds) - train_split])

    train_ds = SubDataset(
        subset = train_subset,
        transform = train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = True,
    )

    val_ds = SubDataset(
        subset = val_subset,
        transform = val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = False,
    )

    return train_loader, val_loader

# Calculates the training metrics for validation dataset, prints them out and returns them
def check_training_metrics(loader, model, loss_fn, device="cuda"):
    accuracy = Accuracy(task="binary").to(device)
    f1 = F1Score(task="binary").to(device)
    iou = JaccardIndex(task="binary").to(device)

    loss = 0
    accuracy_score = 0
    f1_score = 0
    iou_score = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = model(x)

            loss += loss_fn(preds, y)
            accuracy_score += accuracy(preds, y)
            f1_score += f1(preds, y)
            iou_score += iou(preds, y)

    loss = float('{0:.4f}'.format(loss/len(loader)))
    accuracy_score = float('{0:.4f}'.format(accuracy_score/len(loader)*100))
    f1_score = float('{0:.4f}'.format(f1_score/len(loader)))
    iou_score = float('{0:.4f}'.format(iou_score/len(loader)))

    print(f"Validation loss: {loss}")
    print(f"Accuracy reached: {accuracy_score}%")
    print(f"F1 score: {f1_score}")
    print(f"IoU score: {iou_score}")

    model.train()

    return loss, accuracy_score, f1_score, iou_score


# Writes training metrics to the end of a .csv file
def save_training_metrics(file_path, training_metrics):
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(training_metrics)

# Saves the validation dataset images and masks to a folder, so they can be referenced while looking at training segmentation results
def save_val_ds_as_imgs(loader, folder="saved_images/default", device="cuda"):
    os.makedirs(folder, exist_ok = True)
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        torchvision.utils.save_image(x, f"{folder}/orig_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/mask_{idx}.png")

# Saves the validation dataset images and masks to a folder, so they can be referenced while looking at training segmentation results
def save_val_predictions_as_imgs(loader, model, folder="saved_images/default", device="cuda"):
    os.makedirs(folder, exist_ok = True)
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")

    model.train()

# Saves the predicted masks of individual images
def save_predictions_as_imgs(loader, model, folder="saved_images/default", device="cuda"):
    os.makedirs(folder, exist_ok = True)
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{y[0]}.png")

# Parses commands line arguments and returns them
# new models and datasets must be included in here
def parse_args(args):
    selected_model = "UNET"     # model that was selected, must be in VALID_MODELS list constant
    selected_dataset = ""       # dataset that was selected, should be in VALID_DATASETS list constant
    load_model = ""             # path to the checkpoint of the model to be loaded
    num_epochs = 5              # number of epochs to train the network for
    source_dir = ""             # path to the directory with images to be predicted
    early_stop = False          # whether to use early stopping mechanism during training

    for i, arg in enumerate(args):
        if "-m" == arg:
            model_choice = args[i+1]
            if model_choice in VALID_MODELS:
                selected_model = model_choice
            else:
                if model_choice == "1":
                    selected_model = VALID_MODELS[0]
                elif model_choice == "2":
                    selected_model = VALID_MODELS[1]
                elif model_choice == "3":
                    selected_model = VALID_MODELS[2]
                else:
                    print(f"{model_choice} is not a valid model")
                    exit(1)
        elif "-d" == arg:
            dataset_choice = args[i+1]
            if dataset_choice in VALID_DATASETS:
                selected_dataset = dataset_choice
            else:
                if dataset_choice == "1":
                    selected_dataset = VALID_DATASETS[0]
                elif dataset_choice == "2":
                    selected_dataset = VALID_DATASETS[1]
                else:
                    print(f"{dataset_choice} is not a valid dataset")
                    exit(1)
        elif "-l" == arg:
            if os.path.isfile(args[i+1]):
                load_model = args[i+1]
            else:
                print(f"File {args[i+1]} does not exist")
                exit(1)
        elif "-e" == arg:
            if str.isdigit(args[i+1]):
                num_epochs = int(args[i+1])
            else:
                print(f"Invalid value for epochs")
                exit(1)
        elif "-s" == arg:
            if os.path.isdir(args[i+1]):
                source_dir = args[i+1]
            else:
                print(f"Folder {args[i+1]} does not exist")
                exit(1)
        elif "-es" == arg:
            early_stop = True

    #print(f"Selected model: {selected_model}\nSelected dataset: {selected_dataset}\nNumber of epochs: {num_epochs}\nLoad model: {load_model}")
    return selected_model, selected_dataset, load_model, num_epochs, early_stop, source_dir