#!/usr/bin/env python3

"""Create the initial global model checkpoint.

Supports DenseNet-121 and ResNet-50 architectures with configurable
number of classes and input channels. Uses torchvision models.
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_model(arch_config):
    """Create a model from architecture configuration.

    Args:
        arch_config: dict with keys like model, num_classes, in_channels, pretrained

    Returns:
        torch.nn.Module
    """
    import torch
    import torchvision.models as models

    model_name = arch_config.get("model", "DenseNet121").lower()
    num_classes = arch_config.get("num_classes", 14)
    in_channels = arch_config.get("in_channels", 1)
    pretrained = arch_config.get("pretrained", True)

    logger.info(f"Creating model: {model_name}, classes={num_classes}, "
                f"channels={in_channels}, pretrained={pretrained}")

    if "densenet" in model_name:
        if pretrained:
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            model = models.densenet121(weights=None)
        # Adjust first conv for grayscale input
        if in_channels == 1:
            conv0 = model.features.conv0
            model.features.conv0 = torch.nn.Conv2d(
                1, conv0.out_channels,
                kernel_size=conv0.kernel_size,
                stride=conv0.stride,
                padding=conv0.padding,
                bias=False,
            )
            if pretrained:
                # Average RGB weights to initialize grayscale conv
                with torch.no_grad():
                    model.features.conv0.weight.copy_(
                        conv0.weight.mean(dim=1, keepdim=True)
                    )
        # Adjust classifier for num_classes
        num_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_features, num_classes)

    elif "resnet" in model_name:
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            model = models.resnet50(weights=None)
        if in_channels == 1:
            conv1 = model.conv1
            model.conv1 = torch.nn.Conv2d(
                1, conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    model.conv1.weight.copy_(
                        conv1.weight.mean(dim=1, keepdim=True)
                    )
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Create the initial global model checkpoint."
    )
    parser.add_argument("--arch-config", default=None, help="JSON file with model architecture params")
    parser.add_argument("--output-model", required=True, help="Output model checkpoint (.pt)")
    parser.add_argument("--output-config", required=True, help="Output model architecture JSON")

    args = parser.parse_args()

    logger.info(f"Arch config: {args.arch_config}")
    logger.info(f"Output model: {args.output_model}")
    logger.info(f"Output config: {args.output_config}")

    out_model_dir = os.path.dirname(args.output_model)
    if out_model_dir:
        os.makedirs(out_model_dir, exist_ok=True)

    out_config_dir = os.path.dirname(args.output_config)
    if out_config_dir:
        os.makedirs(out_config_dir, exist_ok=True)

    # Load or default architecture config
    if args.arch_config and os.path.exists(args.arch_config):
        with open(args.arch_config, "r") as f:
            arch = json.load(f)
        logger.info(f"Loaded arch config: {arch}")
    else:
        arch = {
            "model": "DenseNet121",
            "num_classes": 14,
            "in_channels": 1,
            "pretrained": True,
        }
        logger.info(f"Using default arch config: {arch}")

    try:
        import torch
        model = create_model(arch)
        torch.save(model.state_dict(), args.output_model)
        logger.info(f"Model state_dict saved: {args.output_model}")
    except ImportError:
        logger.warning("torch not available; writing placeholder checkpoint")
        import pickle
        with open(args.output_model, "wb") as f:
            pickle.dump({"state": "placeholder", "arch": arch}, f)

    # Save architecture metadata
    with open(args.output_config, "w") as f:
        json.dump(arch, f, indent=2)
    logger.info(f"Config saved: {args.output_config}")
    logger.info("initialize_model complete.")


if __name__ == "__main__":
    main()
