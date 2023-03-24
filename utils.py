import torch
import torchvision
import os
from dataset import CarvanaDataset, SubDataset, IMCDBDataset
from torch.utils.data import DataLoader, random_split

VALID_MODELS = [
    "UNET",
    "DoubleUNET",
    "ResUNETpp"
]

VALID_DATASETS = [
    "Carvana",
    "IMCDB"
]

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_carvana_loaders(image_dir, mask_dir, batch_size, train_transform, val_transform, num_workers=4, pin_memory=True):
    init_ds = CarvanaDataset(image_dir, mask_dir)

    train_split = int(len(init_ds)*0.9)
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

def get_imcdb_loaders(data_dir, batch_size, train_transform, val_transform, num_workers=4, pin_memory=True):
    init_ds = IMCDBDataset(data_dir = data_dir)

    train_split = int(len(init_ds)*0.9)
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

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_val_predictions_as_imgs(loader, model, folder="saved_images/default", device="cuda"):
    os.makedirs(folder, exist_ok = True)
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/default", device="cuda"):
    os.makedirs(folder, exist_ok = True)
    model.eval()
    for idx, x in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )

def parse_args(args):
    selected_model = "UNET"
    selected_dataset = ""
    load_model = ""
    num_epochs = 3
    source_dir = ""

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
                    print(f"{model_choice} is not a valid model, defaulted to UNET")
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
            load_model = args[i+1]
        elif "-e" == arg:
            num_epochs = int(args[i+1])
        elif "-s" == arg:
            source_dir = args[i+1]

    print(f"Selected model: {selected_model}\nSelected dataset: {selected_dataset}\nNumber of epochs: {num_epochs}\nLoad model: {load_model}")
    return selected_model, selected_dataset, load_model, num_epochs, source_dir