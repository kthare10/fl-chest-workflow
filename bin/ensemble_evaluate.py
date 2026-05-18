#!/usr/bin/env python3

"""Ensemble evaluation: combine predictions from multiple models.

Methods:
- average: simple unweighted average of sigmoid probabilities
- weighted_average: weight by round number (later rounds get more weight)
- max: element-wise maximum across models

Output: ensemble metrics JSON compatible with evaluate.py format.
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
            logger.info(f"Loaded model weights from {model_path}")
        else:
            logger.warning(f"Model checkpoint is placeholder: {model_path}")
    else:
        logger.warning(f"Model file not found: {model_path}")

    model.to(device)
    model.eval()
    return model


def compute_ensemble_predictions(models, dataloader, device, strategy="average"):
    """Run ensemble inference and aggregate predictions.

    Returns:
        y_true: (N, C) numpy array
        y_ensemble: (N, C) numpy array of aggregated probabilities
    """
    import torch
    import numpy as np

    all_labels = []
    all_preds_per_model = [[] for _ in models]

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(dataloader):
            X = X.to(device)
            all_labels.append(y.numpy())

            for m_idx, model in enumerate(models):
                outputs = model(X)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds_per_model[m_idx].append(probs)

    y_true = np.vstack(all_labels)

    # Stack predictions: (num_models, N, C)
    preds_stack = np.stack([
        np.vstack(preds) for preds in all_preds_per_model
    ], axis=0)

    if strategy == "average":
        y_ensemble = np.mean(preds_stack, axis=0)
    elif strategy == "weighted_average":
        # Weight by round index (if round_1, round_2, ...)
        weights = np.arange(1, len(models) + 1, dtype=np.float32)
        weights = weights / weights.sum()
        weights = weights.reshape(-1, 1, 1)
        y_ensemble = np.sum(preds_stack * weights, axis=0)
    elif strategy == "max":
        y_ensemble = np.max(preds_stack, axis=0)
    else:
        raise ValueError(f"Unknown ensemble strategy: {strategy}")

    return y_true, y_ensemble


def evaluate_ensemble(y_true, y_pred, num_classes=14):
    """Compute per-class metrics for ensemble predictions."""
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

    y_binary = (y_pred >= 0.5).astype(int)

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
        description="Ensemble evaluation combining multiple FL models."
    )
    parser.add_argument("--models", nargs="+", required=True, help="Model checkpoint files (.pt)")
    parser.add_argument("--data-dir", required=True, help="Test data directory")
    parser.add_argument("--dataset-name", choices=["tcia", "nih"], required=True)
    parser.add_argument("--output-metrics", required=True, help="Output ensemble metrics JSON")
    parser.add_argument("--arch-config", default=None, help="Model architecture JSON")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument(
        "--ensemble-strategy",
        choices=["average", "weighted_average", "max"],
        default="average",
        help="Ensemble aggregation strategy",
    )

    args = parser.parse_args()

    logger.info(f"Models:         {args.models}")
    logger.info(f"Data dir:       {args.data_dir}")
    logger.info(f"Dataset:        {args.dataset_name}")
    logger.info(f"Strategy:       {args.ensemble_strategy}")
    logger.info(f"Device:         {args.device}")

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
        metrics = {"dataset": args.dataset_name, "status": "no_torch", "ensemble": True}
        with open(args.output_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        return

    # Load architecture config
    if args.arch_config and os.path.exists(args.arch_config):
        with open(args.arch_config, "r") as f:
            arch = json.load(f)
    else:
        default_arch = os.path.join(os.path.dirname(args.models[0]), "model_arch.json")
        if os.path.exists(default_arch):
            with open(default_arch, "r") as f:
                arch = json.load(f)
        else:
            arch = {"model": "DenseNet121", "num_classes": 14, "in_channels": 1}

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load all models
    models = []
    for model_path in args.models:
        model = load_model(model_path, arch, device)
        models.append(model)

    logger.info(f"Loaded {len(models)} models for ensemble")

    # Load test data (reuse evaluate.py logic)
    labels_file = os.path.join(args.data_dir, "labels.json")
    from evaluate import load_test_data
    dataset = load_test_data(args.data_dir, labels_file, dataset_name=args.dataset_name)
    if dataset is None:
        logger.info("Using synthetic test data")
        from evaluate import create_synthetic_testset
        dataset = create_synthetic_testset(num_samples=30)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    # Ensemble inference
    y_true, y_pred = compute_ensemble_predictions(
        models, dataloader, device, strategy=args.ensemble_strategy
    )

    metrics = evaluate_ensemble(y_true, y_pred, num_classes=arch.get("num_classes", 14))
    metrics["dataset"] = args.dataset_name
    metrics["status"] = "ok"
    metrics["ensemble"] = True
    metrics["ensemble_strategy"] = args.ensemble_strategy
    metrics["num_models"] = len(models)

    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Ensemble metrics saved: {args.output_metrics}")
    logger.info(f"  Macro AUC: {metrics['macro']['auc']:.4f}")
    logger.info(f"  Macro F1:  {metrics['macro']['f1']:.4f}")
    logger.info("ensemble_evaluate complete.")


if __name__ == "__main__":
    main()
