import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

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

    # Make results reproducible.
    torch.manual_seed(42)

    # Generate shuffled indices.
    indices = torch.randperm(len(train_dataset)).tolist()

    train_size = int(0.8 * len(indices))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

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