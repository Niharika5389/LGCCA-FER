from typing import Tuple

import torch
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


class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 (timm) classification head with a reusable embedding.

    Feature extraction is delegated to
    ``timm.create_model('efficientnet_b0', pretrained=True, num_classes=0,
    global_pool='avg')``, which yields a 1280-d pooled embedding. That
    embedding is passed through a small MLP head (Linear(1280,256) ->
    ReLU -> Dropout(0.4) -> Linear(256, num_classes)) for classification.

    Unlike ``EfficientNetBackbone`` above (kept for backward compatibility
    with existing baseline notebooks), ``forward`` returns both the logits
    and the pooled embedding, so later modules (e.g. IAFM, LGCCA) can reuse
    the embedding without changing this class's return signature.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        embedding_dim: int = 1280,
        hidden_dim: int = 256,
        dropout: float = 0.4,
        pretrained: bool = True,
    ) -> None:
        """
        Args:
            num_classes: Number of output emotion classes.
            embedding_dim: Dimensionality of the pooled backbone feature
                (1280 for EfficientNet-B0).
            hidden_dim: Hidden layer size of the classification head.
            dropout: Dropout probability applied before the final linear
                layer.
            pretrained: Whether to load ImageNet-pretrained backbone
                weights.
        """
        super().__init__()

        import timm  # local import: keeps torchvision-only path lightweight

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the backbone + classification head.

        Args:
            x: Input image batch, shape ``[B, 3, H, W]``.

        Returns:
            A tuple ``(logits, pooled_embedding)`` where ``logits`` has
            shape ``[B, num_classes]`` and ``pooled_embedding`` has shape
            ``[B, embedding_dim]``.
        """
        pooled_embedding = self.backbone(x)
        logits = self.classifier(pooled_embedding)
        return logits, pooled_embedding