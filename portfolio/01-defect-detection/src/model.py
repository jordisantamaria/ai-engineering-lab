"""
Defect detection model using EfficientNet-B0 with transfer learning.

This module provides a classifier that leverages a pretrained EfficientNet-B0
backbone and replaces the final fully-connected layer for binary or multi-class
defect classification.
"""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class DefectClassifier(nn.Module):
    """
    Defect classifier built on top of EfficientNet-B0.

    Uses transfer learning: the pretrained backbone extracts visual features
    while a custom classification head maps them to defect categories.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.3,
    ):
        """
        Initialize the defect classifier.

        Args:
            num_classes: Number of output classes (2 for binary: good/defect).
            pretrained: Whether to load ImageNet pretrained weights.
            freeze_backbone: Whether to freeze the backbone layers.
            dropout_rate: Dropout probability before the final linear layer.
        """
        super().__init__()

        # Load the pretrained EfficientNet-B0 backbone
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        return self.backbone(x)

    def unfreeze_backbone(self, num_layers: int = -1) -> None:
        """
        Unfreeze backbone layers for fine-tuning.

        Args:
            num_layers: Number of layers to unfreeze from the end.
                       Use -1 to unfreeze all layers.
        """
        params = list(self.backbone.features.parameters())
        if num_layers == -1:
            for param in params:
                param.requires_grad = True
        else:
            for param in params[-num_layers:]:
                param.requires_grad = True


def get_inference_transform() -> transforms.Compose:
    """
    Get the standard image transform for inference.

    Returns:
        A torchvision Compose transform that resizes, normalizes, and
        converts an image to a tensor suitable for EfficientNet-B0.
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_model(
    checkpoint_path: str,
    num_classes: int = 2,
    device: Optional[str] = None,
) -> Tuple[DefectClassifier, torch.device]:
    """
    Load a trained defect classifier from a checkpoint file.

    Args:
        checkpoint_path: Path to the saved model weights (.pth file).
        num_classes: Number of classes the model was trained on.
        device: Device string ('cuda', 'cpu'). Auto-detected if None.

    Returns:
        A tuple of (model, device) with the model in eval mode.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = DefectClassifier(
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both raw state_dict and checkpoint dict formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, device


def predict_image(
    model: DefectClassifier,
    image_path: str,
    transform: Optional[transforms.Compose] = None,
    device: Optional[torch.device] = None,
    class_names: Optional[list] = None,
) -> Dict:
    """
    Run inference on a single image.

    Args:
        model: Trained DefectClassifier in eval mode.
        image_path: Path to the image file.
        transform: Image preprocessing transform. Uses default if None.
        device: Torch device. Uses CPU if None.
        class_names: List of class names. Defaults to ['good', 'defect'].

    Returns:
        Dictionary with keys:
            - 'class': Predicted class name.
            - 'class_index': Predicted class index.
            - 'confidence': Confidence score (0-1) for the predicted class.
            - 'probabilities': Dict mapping class names to their probabilities.
            - 'inference_time_ms': Inference time in milliseconds.
    """
    if transform is None:
        transform = get_inference_transform()
    if device is None:
        device = next(model.parameters()).device
    if class_names is None:
        class_names = ["good", "defect"]

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference with timing
    start_time = time.time()
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
    inference_time = (time.time() - start_time) * 1000  # Convert to ms

    # Extract predictions
    confidence, predicted_idx = torch.max(probabilities, dim=1)
    predicted_idx = predicted_idx.item()
    confidence = confidence.item()

    # Build probability mapping
    prob_dict = {
        class_names[i]: round(probabilities[0][i].item(), 4)
        for i in range(len(class_names))
    }

    return {
        "class": class_names[predicted_idx],
        "class_index": predicted_idx,
        "confidence": round(confidence, 4),
        "probabilities": prob_dict,
        "inference_time_ms": round(inference_time, 2),
    }
