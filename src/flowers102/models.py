from __future__ import annotations

from typing import Literal

import torch.nn as nn
import torchvision.models as tvm


ModelName = Literal[
    "resnet18", "resnet50",
    "efficientnet_b0", "efficientnet_b3",
    "vgg16", "convnext_tiny", "swin_t", "vit_b_16",
]


def _replace_classifier(model: nn.Module, model_name: str, num_classes: int) -> nn.Module:
    if model_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name.startswith("vgg"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name.startswith("convnext"):
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif model_name.startswith("swin"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    elif model_name.startswith("vit"):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model


def create_model(
    model_name: ModelName,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    weights = "DEFAULT" if pretrained else None

    if model_name == "resnet18":
        model = tvm.resnet18(weights=weights)
    elif model_name == "resnet50":
        model = tvm.resnet50(weights=weights)
    elif model_name == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=weights)
    elif model_name == "efficientnet_b3":
        model = tvm.efficientnet_b3(weights=weights)
    elif model_name == "vgg16":
        model = tvm.vgg16(weights=weights)
    elif model_name == "convnext_tiny":
        model = tvm.convnext_tiny(weights=weights)
    elif model_name == "swin_t":
        model = tvm.swin_t(weights=weights)
    elif model_name == "vit_b_16":
        model = tvm.vit_b_16(weights=weights)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return _replace_classifier(model, model_name, num_classes)


def freeze_feature_extractor(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, "heads"):
        for param in model.heads.parameters():
            param.requires_grad = True
    elif hasattr(model, "head"):
        for param in model.head.parameters():
            param.requires_grad = True
    elif hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True

