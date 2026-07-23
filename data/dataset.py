from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from configs.config import DATASET_PATH, BATCH_SIZE
from data.transforms import train_transform, val_transform


def get_dataloaders():
    """
    Creates DataLoaders for training, validation and testing.
    """

    # Load the complete training dataset.
    # At this stage, every image uses the training transforms.
    full_train_dataset = ImageFolder(
        root=f"{DATASET_PATH}/train",
        transform=train_transform
    )

    # Load the test dataset.
    # Test images should never be randomly augmented.
    test_dataset = ImageFolder(
        root=f"{DATASET_PATH}/test",
        transform=val_transform
    )

    # Calculate how many images go into training and validation.
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # Split the training dataset.
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size]
    )

    # Validation should not use random augmentation.
    val_dataset.dataset.transform = val_transform

    # Create DataLoaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader