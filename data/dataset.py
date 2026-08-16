import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

from configs.config import DATASET_PATH, BATCH_SIZE
from data.transforms import train_transform, val_transform


def get_dataloaders():
    """
    Creates DataLoaders for training, validation and testing.
    """

    # Load the same training folder twice.
    # One copy will use training augmentation.
    # The other copy will use validation transforms.
    train_dataset = ImageFolder(
        root=f"{DATASET_PATH}/train",
        transform=train_transform
    )

    val_dataset = ImageFolder(
        root=f"{DATASET_PATH}/train",
        transform=val_transform
    )

    test_dataset = ImageFolder(
        root=f"{DATASET_PATH}/test",
        transform=val_transform
    )

    # Get the label for every image, so the split can respect
    # class proportions instead of a plain random cut.
    labels = [label for _, label in train_dataset.samples]

    # Stratified split: each class keeps the same 80/20 ratio
    # between train and val.
    train_indices, val_indices = train_test_split(
        range(len(train_dataset)),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # Create subsets.
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader