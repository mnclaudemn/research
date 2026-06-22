import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(image_size=224):

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    test_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    return train_tf, test_tf


def find_folder(dataset_path, name_options):
    """
    Find folder like train/test/val automatically
    """
    for d in os.listdir(dataset_path):
        if d.lower() in name_options:
            return os.path.join(dataset_path, d)
    return None


def get_loaders(dataset_path, batch_size=16, image_size=224):

    train_tf, test_tf = get_transforms(image_size)

    #  Auto-detect folders
    train_dir = find_folder(dataset_path, ["train", "training"])
    test_dir = find_folder(dataset_path, ["test", "testing"])
    val_dir = find_folder(dataset_path, ["val", "valid", "validation"])

    if train_dir is None:
        raise ValueError("Train folder not found!")

    #  datasets
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)

    val_ds = datasets.ImageFolder(val_dir, transform=test_tf) if val_dir else None
    test_ds = datasets.ImageFolder(test_dir, transform=test_tf) if test_dir else None

    # loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size) if test_ds else None

    return train_loader, val_loader, test_loader, len(train_ds.classes)
