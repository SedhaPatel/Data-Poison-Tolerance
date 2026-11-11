"""
Model Architectures for Fashion MNIST Pipeline
Contains EnhancedCNN (fine-tuned high performance) and BasicCNN (baseline architecture),
along with builder and model loading utilities.
"""

import os
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

class BasicCNN(nn.Module):
    """Original 2-block Convolutional Neural Network baseline."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10, dropout_rate: float = 0.3):
        super(BasicCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 1x28x28 -> 32x14x14
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32x14x14 -> 64x7x7
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

class EnhancedCNN(nn.Module):
    """
    Fine-Tuned 3-Stage Convolutional Neural Network.
    Optimized for Fashion MNIST classification to achieve maximum accuracy (>94%).
    Features:
      - Double-convolution initial feature extractors with Batch Normalization
      - Progressive channel depths: 32 -> 64 -> 128
      - Multi-stage regularized dropout
      - High-capacity dense classifier with Batch Normalization
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        conv_channels: Optional[list] = None,
        fc_units: int = 256,
        dropout_rate: float = 0.35,
        use_batch_norm: bool = True,
        activation: str = "relu"
    ):
        super(EnhancedCNN, self).__init__()

        if conv_channels is None:
            conv_channels = [32, 64, 128]

        c1, c2, c3 = conv_channels[0], conv_channels[1], conv_channels[2]

        act_layer = nn.ReLU if activation.lower() == "relu" else (
            nn.LeakyReLU if activation.lower() == "leaky_relu" else nn.GELU
        )

        def make_bn(channels: int):
            return nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()

        # Stage 1: 1x28x28 -> c1x28x28 -> c1x14x14
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=not use_batch_norm),
            make_bn(c1),
            act_layer(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=not use_batch_norm),
            make_bn(c1),
            act_layer(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.15)
        )

        # Stage 2: c1x14x14 -> c2x14x14 -> c2x7x7
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=not use_batch_norm),
            make_bn(c2),
            act_layer(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=not use_batch_norm),
            make_bn(c2),
            act_layer(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.20)
        )

        # Stage 3: c2x7x7 -> c3x7x7 -> c3x3x3
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=not use_batch_norm),
            make_bn(c3),
            act_layer(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )

        # Feature map size at stage 3 output: 3 x 3
        flattened_dim = c3 * 3 * 3

        # Classifier head
        classifier_layers = [
            nn.Flatten(),
            nn.Linear(flattened_dim, fc_units, bias=not use_batch_norm),
        ]
        if use_batch_norm:
            classifier_layers.append(nn.BatchNorm1d(fc_units))
        classifier_layers.extend([
            act_layer(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_units, num_classes)
        ])
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.classifier(x)
        return x

def build_cnn_model(cnn_config: Dict[str, Any]) -> nn.Module:
    """Factory function to build the CNN model from configuration dictionary."""
    arch_type = cnn_config.get("architecture", "EnhancedCNN")

    if arch_type == "BasicCNN":
        model = BasicCNN(
            in_channels=cnn_config.get("in_channels", 1),
            num_classes=cnn_config.get("num_classes", 10),
            dropout_rate=cnn_config.get("dropout_rate", 0.3)
        )
    else:
        model = EnhancedCNN(
            in_channels=cnn_config.get("in_channels", 1),
            num_classes=cnn_config.get("num_classes", 10),
            conv_channels=cnn_config.get("conv_channels", [32, 64, 128]),
            fc_units=cnn_config.get("fc_units", 256),
            dropout_rate=cnn_config.get("dropout_rate", 0.35),
            use_batch_norm=cnn_config.get("use_batch_norm", True),
            activation=cnn_config.get("activation", "relu")
        )

    return model

def load_cnn_checkpoint(model_path: str, cnn_config: Optional[Dict[str, Any]] = None, device: str = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Intelligently loads a CNN checkpoint from disk.
    Supports both new checkpoints (dict with config + state_dict)
    and raw state_dict files (detecting architecture automatically).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    meta = {}

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # Checkpoint bundle
        state_dict = checkpoint["state_dict"]
        arch_config = checkpoint.get("hyperparameters", cnn_config or {})
        meta = {k: v for k, v in checkpoint.items() if k != "state_dict"}
        model = build_cnn_model(arch_config)
        model.load_state_dict(state_dict)
    elif isinstance(checkpoint, dict):
        # Raw state_dict
        state_dict = checkpoint
        # Auto-detect architecture by keys
        if any(k.startswith("stage1") for k in state_dict.keys()):
            if cnn_config is None:
                cnn_config = {"architecture": "EnhancedCNN"}
            model = build_cnn_model(cnn_config)
        else:
            model = BasicCNN()
        model.load_state_dict(state_dict)
    else:
        # Full model object
        model = checkpoint

    model.to(device)
    model.eval()
    return model, meta
