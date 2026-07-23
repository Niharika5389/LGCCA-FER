from torchvision import transforms

from configs.config import IMAGE_SIZE


# These transforms are applied only while training.
# Random augmentations help the model learn better and reduce overfitting.

train_transform = transforms.Compose([

    # Resize every image to 224 × 224 for EfficientNet-B0
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # Randomly flip faces horizontally
    transforms.RandomHorizontalFlip(),

    # Slightly rotate the image
    # This helps the model handle tilted faces.
    transforms.RandomRotation(10),

    # Convert image into a PyTorch Tensor
    transforms.ToTensor(),

    # Normalize using ImageNet statistics.
    # EfficientNet was pretrained on ImageNet, so using the same normalization improves performance.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation images should never be randomly changed.
# We only resize and normalize them.

val_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Testing should be identical to validation.

test_transform = val_transform