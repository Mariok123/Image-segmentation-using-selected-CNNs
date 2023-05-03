import torch
import torchvision
import os
import csv
from dataset import GenericDataset, SubDataset, IMCDBDataset
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
    "data/Carvana/train/",
    "data/Carvana/train_masks/"
]

IMCDB_DIR = "data/IMCDB/"

# Saves the checkpoint of the currently training model
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)

# Loads the checkpoint of a previously trained model
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


# Returns necessary items that are unique between the different implemented models
# if another model was to be implemented, it also has to be added here
def get_model(selected_model, device="cuda", learning_rate = 1e-4):
    if selected_model == "UNET":
        model = UNET().to(device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, nesterov=True, momentum=0.99)
    elif selected_model == "DoubleUNET":
        model = DoubleUNET().to(device)

        loss_fn = DiceLoss()                                          # one of the two combinations may perform slightly better on certain datasets
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  #

        #loss_fn = nn.BCEWithLogitsLoss()
        #optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
    elif selected_model == "ResUNETpp":
        model = ResUNETpp().to(device)
        loss_fn = DiceLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        print("Unknown model")
        exit(1)

    return model, loss_fn, optimizer

# Returns the training and validation dataloader for one of the supported datasets
# if another dataset was to be added, it also has to be added here
def get_loaders(selected_dataset, batch_size, train_transform, val_transform, num_workers, pin_memory, dataset_split=0.8, ds_source_dir="", ds_mask_dir=""):
    if selected_dataset == "Carvana":
        init_ds = GenericDataset(CARVANA_DIR[0], CARVANA_DIR[1])
    elif selected_dataset == "IMCDB":
        init_ds = IMCDBDataset(IMCDB_DIR)
    elif ds_source_dir and ds_mask_dir:
        init_ds = GenericDataset(ds_source_dir, ds_mask_dir)
    else:
        print("Unknown dataset information")
        exit(1)
    
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

# Validates model, calculates and prints training metrics and saves segmented masks if chosen to
def validate_model(loader, model, loss_fn, save_validation_results=False, folder="training_results/default", device="cuda"):
    accuracy = Accuracy(task="binary").to(device)
    f1 = F1Score(task="binary").to(device)
    iou = JaccardIndex(task="binary").to(device)

    val_loss = 0
    accuracy_score = 0
    f1_score = 0
    iou_score = 0

    model.eval()

    if save_validation_results:
        os.makedirs(folder, exist_ok = True)

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = model(x)

            val_loss += loss_fn(preds, y)
            accuracy_score += accuracy(preds, y)
            f1_score += f1(preds, y)
            iou_score += iou(preds, y)

            if save_validation_results:
                preds = torch.sigmoid(preds)
                preds = (preds > 0.5).float()
                torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")

    val_loss = float('{0:.4f}'.format(val_loss/len(loader)))
    accuracy_score = float('{0:.4f}'.format(accuracy_score/len(loader)*100))
    f1_score = float('{0:.4f}'.format(f1_score/len(loader)))
    iou_score = float('{0:.4f}'.format(iou_score/len(loader)))

    print(f"Validation loss: {val_loss}")
    print(f"Accuracy reached: {accuracy_score}%")
    print(f"F1 score: {f1_score}")
    print(f"IoU score: {iou_score}")

    model.train()

    return val_loss, accuracy_score, f1_score, iou_score

# Writes training metrics to the end of a .csv file
def save_training_metrics(folder, training_metrics):
    os.makedirs(folder, exist_ok = True)
    with open(folder + "training_metrics.csv", 'a', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(training_metrics)

# Saves the validation dataset images and masks to a folder, so they can be referenced while looking at training segmentation results
def save_val_ds_as_imgs(loader, folder="saved_images/default", device="cuda"):
    os.makedirs(folder, exist_ok = True)
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        torchvision.utils.save_image(x, f"{folder}/orig_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/mask_{idx}.png")

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

# Parses commands line arguments used for training
def parse_training_args(args):
    selected_model = ""
    selected_dataset = ""
    dataset_source_dir = ""
    dataset_mask_dir = ""
    num_epochs = 5
    checkpoint_path = ""

    model_arg, dataset_arg, dataset_source_arg, dataset_mask_arg, epoch_arg, load_model_arg, _, early_stop, save_validation_results = parse_args(args)

    if model_arg:
        if model_arg in VALID_MODELS:
            selected_model = model_arg
        elif str.isdigit(model_arg):
            if (model_arg := int(model_arg)) in range(1, len(VALID_MODELS)+1):
                selected_model = VALID_MODELS[model_arg-1]

    if dataset_arg:
        if dataset_arg in VALID_DATASETS:
            selected_dataset = dataset_arg
        elif str.isdigit(dataset_arg):
            if (dataset_arg := int(dataset_arg)) in range(1, len(VALID_DATASETS)+1):
                selected_dataset = VALID_DATASETS[dataset_arg-1]
    
    if dataset_source_arg:
        if os.path.isdir(dataset_source_arg):
            dataset_source_dir = dataset_source_arg

    if dataset_mask_arg:
        if os.path.isdir(dataset_mask_arg):
            dataset_mask_dir = dataset_mask_arg

    if epoch_arg:
        if str.isdigit(epoch_arg):
            num_epochs = int(epoch_arg)
        
    if load_model_arg:
        if os.path.isfile(load_model_arg):
            checkpoint_path = load_model_arg
        else:
            print("Invalid -l argument")
            exit(1)

    if not selected_model:
        print("Invalid or missing -m argument")
        exit(1)
    
    if not selected_dataset:
        if (dataset_source_dir and not dataset_mask_dir) or (not dataset_source_dir and dataset_mask_dir):
            print("Invalid or missing -ds or -dm argument")
            exit(1)
        elif not dataset_source_dir and not dataset_mask_dir:
            print("Invalid or missing -d argument")
            exit(1)
    else:
        if selected_dataset == "Carvana":
            for dir in CARVANA_DIR:
                if not os.path.isdir(dir):
                    print(f"Missing source folder {dir} for Carvana dataset")
                    exit(1)
        elif selected_dataset == "IMCDB":
            if not os.path.isdir(IMCDB_DIR):
                print(f"Missing source folder {IMCDB_DIR} for IMCDB dataset")
                exit(1)
    
    #print(f"Selected model: {selected_model}\nSelected dataset: {selected_dataset}\nNumber of epochs: {num_epochs}")
    return selected_model, selected_dataset, dataset_source_dir, dataset_mask_dir, num_epochs, checkpoint_path, early_stop, save_validation_results

# Parses commands line arguments used for prediction
def parse_predict_args(args):
    selected_model = ""
    checkpoint_path = ""
    source_dir_path = ""

    model_arg, _, _, _, _, load_model_arg, source_dir_arg, _, _ = parse_args(args)

    if model_arg:
        if model_arg in VALID_MODELS:
            selected_model = model_arg
        elif str.isdigit(model_arg):
            if (model_arg := int(model_arg)) in range(1, len(VALID_MODELS)+1):
                selected_model = VALID_MODELS[model_arg-1]

    if load_model_arg:
        if os.path.isfile(load_model_arg):
            checkpoint_path = load_model_arg

    if source_dir_arg:
        if os.path.isdir(source_dir_arg):
            source_dir_path = source_dir_arg

    if not selected_model:
        print("Invalid or missing -m argument")
        exit(1)

    if not checkpoint_path:
        print("Invalid or missing -l argument")
        exit(1)

    if not source_dir_path:
        print("Invalid or missing -s argument")
        exit(1)

    return selected_model, checkpoint_path, source_dir_path

# Parses commands line arguments, but does not check them
def parse_args(args):
    model_arg = ""
    dataset_arg = ""
    dataset_source_arg = ""
    dataset_mask_arg = ""
    epoch_arg = ""
    load_model_arg = ""
    source_dir_arg = ""
    early_stop = False
    save_validation_results = False

    if len(args) < 2:
        printHelp()
        exit(0)

    for i, arg in enumerate(args):
        if "-h" == arg:
            printHelp()
            exit(0)
        elif "-m" == arg:
            model_arg = args[i+1]
        elif "-d" == arg:
            dataset_arg = args[i+1]
        elif "-ds" == arg:
            dataset_source_arg = args[i+1]
        elif "-dm" == arg:
            dataset_mask_arg = args[i+1]
        elif "-e" == arg:
            epoch_arg = args[i+1]
        elif "-l" == arg:
            load_model_arg = args[i+1]
        elif "-s" == arg:
            source_dir_arg = args[i+1]
        elif "-es" == arg:
            early_stop = True
        elif "-sv" == arg:
            save_validation_results = True

    return model_arg, dataset_arg, dataset_source_arg, dataset_mask_arg, epoch_arg, load_model_arg, source_dir_arg, early_stop, save_validation_results

# Prints help info
def printHelp():
    print("Usage:")
    print(" train.py -m MODEL -d DATASET [-l PATH] [-e INT] [-sv] [-st]")
    print(" train.py -m MODEL -ds PATH -dm PATH [-l PATH] [-e INT] [-sv] [-st]")
    print(" predict.py -m MODEL -l PATH -s PATH")
    print("")

    print("Model arguments:")
    print(f"{' -m MODEL':10s}\tmodel to use for training/prediction, accepts name or numbers (1-{len(VALID_MODELS)}), {VALID_MODELS}")
    print(f"{' -l PATH':10s}\tload model checkpoint from path")
    print("")

    print("Data arguments")
    print(f"{' -d DATASET':10s}\tdataset to use for training, accepts name or numbers (1-{len(VALID_DATASETS)}), can be substituted with -ds and -dm (if both used, -d takes priority), {VALID_DATASETS}")
    print(f"{' -ds PATH':10s}\tpath to dataset image directory")
    print(f"{' -dm PATH':10s}\tpath to dataset mask directory")
    print(f"{' -s PATH':10s}\tpath to directory of images to predict")
    print("")

    print("Optional arguments:")
    print(f"{' -h ':10s}\tprints out help and exits")
    print(f"{' -es ':10s}\tenable early stopping for training (default False)")
    print(f"{' -sv ':10s}\tsave segmentation masks produced during validation (default False)")
    print(f"{' -e INT':10s}\thow many epochs to train model for (default 5)")