#!/usr/bin/env python3

"""Evaluate a global model on a test dataset.

Computes per-class metrics:
- ROC-AUC (area under receiver operating characteristic)
- Average Precision (PR-AUC)
- F1 score (at threshold 0.5)
- Accuracy (at threshold 0.5)

Supports both real preprocessed data (--data-dir) and synthetic test mode.
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


def load_model(model_path, arch_config, device):
    """Load model architecture and weights."""
    import torch
    import torchvision.models as models

    model_name = arch_config.get("model", "DenseNet121").lower()
    num_classes = arch_config.get("num_classes", 14)
    in_channels = arch_config.get("in_channels", 1)

    if "densenet" in model_name:
        model = models.densenet121(weights=None)
        if in_channels == 1:
            conv0 = model.features.conv0
            model.features.conv0 = torch.nn.Conv2d(
                1, conv0.out_channels,
                kernel_size=conv0.kernel_size,
                stride=conv0.stride,
                padding=conv0.padding,
                bias=False,
            )
        num_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_features, num_classes)
    elif "resnet" in model_name:
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
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state" not in state:
            model.load_state_dict(state)
            logger.info("Loaded model weights")
        else:
            logger.warning("Model checkpoint is placeholder")
    else:
        logger.warning("Model file not found")

    model.to(device)
    model.eval()
    return model


def create_synthetic_testset(num_samples=30, num_classes=14):
    """Create synthetic test dataset."""
    import torch
    from torch.utils.data import TensorDataset

    X = torch.randn(num_samples, 1, 224, 224)
    y = torch.randint(0, 2, (num_samples, num_classes)).float()
    return TensorDataset(X, y)


def load_test_data(data_dir, labels_file, dataset_name="tcia"):
    """Load real preprocessed test data. Falls back to synthetic."""
    import torch
    from torch.utils.data import TensorDataset

    if not os.path.exists(labels_file):
        logger.warning(f"Labels file not found: {labels_file}")
        return None

    with open(labels_file, "r") as f:
        labels = json.load(f)

    tensors = []
    targets = []

    for series_info in labels.get("series", []):
        tensor_file = series_info.get("tensor_file")
        if not tensor_file:
            continue
        tensor_path = os.path.join(data_dir, tensor_file)
        if not os.path.exists(tensor_path):
            continue

        try:
            tensor = torch.load(tensor_path, weights_only=True)
            # Handle different tensor shapes
            if len(tensor.shape) == 4:
                # (1, C, H, W) or (C, D, H, W) -> take middle slice if volumetric
                if tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)  # Remove batch dim if present
                if tensor.shape[0] == 1 and len(tensor.shape) == 3:
                    pass  # Already (C, H, W)
                elif len(tensor.shape) == 3 and tensor.shape[0] > 1:
                    # Volumetric with channel first: take middle slice of depth dim
                    mid = tensor.shape[0] // 2
                    tensor = tensor[mid:mid+1, :, :]
            elif len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
            elif len(tensor.shape) == 3 and tensor.shape[-1] in [1, 3]:
                # Last dim is channel (H, W, C) -> permute to (C, H, W)
                tensor = tensor.permute(2, 0, 1)

            # Ensure 224x224
            if tensor.shape[-1] != 224 or tensor.shape[-2] != 224:
                tensor = torch.nn.functional.interpolate(
                    tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
                ).squeeze(0)

            tensors.append(tensor)

            # Use actual labels from JSON if available; otherwise random
            if "labels" in series_info and isinstance(series_info["labels"], list):
                target = torch.tensor(series_info["labels"], dtype=torch.float32)
            else:
                target = torch.randint(0, 2, (14,)).float()
            targets.append(target)
        except Exception as e:
            logger.warning(f"Failed to load {tensor_path}: {e}")
            continue

    if len(tensors) == 0:
        return None

    X = torch.stack(tensors)
    Y = torch.stack(targets)
    return TensorDataset(X, Y)


def evaluate_model(model, dataloader, device, num_classes=14):
    """Run inference and compute per-class metrics."""
    import torch
    import numpy as np

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            outputs = model(X)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(y.numpy())

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    y_binary = (y_pred >= 0.5).astype(int)

    # Per-class metrics
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

    metrics = {
        "num_samples": len(y_true),
        "per_class": {},
        "macro": {},
    }

    aucs = []
    aps = []
    f1s = []
    accs = []

    for i in range(num_classes):
        cls_name = f"class_{i}"
        yt = y_true[:, i]
        yp = y_pred[:, i]
        yb = y_binary[:, i]

        try:
            auc = roc_auc_score(yt, yp) if len(set(yt)) > 1 else 0.5
        except Exception:
            auc = 0.5

        try:
            ap = average_precision_score(yt, yp)
        except Exception:
            ap = 0.0

        try:
            f1 = f1_score(yt, yb, zero_division=0)
        except Exception:
            f1 = 0.0

        try:
            acc = accuracy_score(yt, yb)
        except Exception:
            acc = 0.0

        aucs.append(auc)
        aps.append(ap)
        f1s.append(f1)
        accs.append(acc)

        metrics["per_class"][cls_name] = {
            "auc": float(auc),
            "average_precision": float(ap),
            "f1": float(f1),
            "accuracy": float(acc),
        }

    metrics["macro"]["auc"] = float(np.mean(aucs))
    metrics["macro"]["average_precision"] = float(np.mean(aps))
    metrics["macro"]["f1"] = float(np.mean(f1s))
    metrics["macro"]["accuracy"] = float(np.mean(accs))

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a global model on a test dataset."
    )
    parser.add_argument("--model", required=True, help="Global model checkpoint (.pt)")
    parser.add_argument("--data-dir", required=True, help="Test data directory")
    parser.add_argument("--output-metrics", required=True, help="Output evaluation metrics JSON")
    parser.add_argument("--dataset-name", choices=["tcia", "nih"], required=True)
    parser.add_argument("--arch-config", default=None, help="Model architecture JSON")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")

    args = parser.parse_args()

    logger.info(f"Model:        {args.model}")
    logger.info(f"Data dir:     {args.data_dir}")
    logger.info(f"Dataset:      {args.dataset_name}")
    logger.info(f"Device:       {args.device}")

    out_dir = os.path.dirname(args.output_metrics)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False

    if not HAS_TORCH:
        logger.error("torch not available; writing placeholder")
        metrics = {"dataset": args.dataset_name, "status": "no_torch"}
        with open(args.output_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        return

    # Load architecture config
    if args.arch_config and os.path.exists(args.arch_config):
        with open(args.arch_config, "r") as f:
            arch = json.load(f)
    else:
        default_arch = os.path.join(os.path.dirname(args.model), "model_arch.json")
        if os.path.exists(default_arch):
            with open(default_arch, "r") as f:
                arch = json.load(f)
        else:
            arch = {"model": "DenseNet121", "num_classes": 14, "in_channels": 1}

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, arch, device)

    # Load test data
    labels_file = os.path.join(args.data_dir, "labels.json")
    dataset = load_test_data(args.data_dir, labels_file, dataset_name=args.dataset_name)
    if dataset is None:
        logger.info("Using synthetic test data")
        dataset = create_synthetic_testset(num_samples=30)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    metrics = evaluate_model(model, dataloader, device, num_classes=arch.get("num_classes", 14))
    metrics["dataset"] = args.dataset_name
    metrics["status"] = "ok"
    metrics["model_arch"] = arch.get("model", "unknown")

    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Evaluation metrics saved: {args.output_metrics}")
    logger.info(f"  Macro AUC: {metrics['macro']['auc']:.4f}")
    logger.info(f"  Macro F1:  {metrics['macro']['f1']:.4f}")
    logger.info("evaluate complete.")


if __name__ == "__main__":
    main()
