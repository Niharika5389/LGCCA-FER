import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision.models import EfficientNet_B0_Weights

from configs.config import NUM_CLASSES


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet-B0 model for facial emotion recognition.
    """

    def __init__(self):
        super().__init__()

        # Load the pretrained EfficientNet-B0 model.
        # The pretrained weights help the model learn faster.
        self.model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        # Find the number of input features of the final classifier.
        in_features = self.model.classifier[1].in_features

        # Replace the original ImageNet classifier (1000 classes)
        # with a new classifier for our 7 emotion classes.
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, NUM_CLASSES)
        )

    def forward(self, x):
        """
        Defines how an input image passes through the network.
        """
        return self.model(x)