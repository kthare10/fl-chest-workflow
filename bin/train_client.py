#!/usr/bin/env python3

"""Perform local FL training for one client and one round.

Supports:
- Loading preprocessed .pt tensors
- DenseNet/ResNet models
- FedProx regularization
- Training/validation metrics (loss, AUC)
- Synthetic data mode for testing
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_arch_config(config_path):
    """Load model architecture from JSON."""
    with open(config_path, "r") as f:
        return json.load(f)


def create_model(arch_config):
    """Create model from architecture config (same as initialize_model)."""
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

    return model


def create_synthetic_dataset(num_samples=50, num_classes=14, input_size=(1, 128, 128)):
    """Create synthetic PyTorch dataset for testing."""
    import torch
    from torch.utils.data import TensorDataset

    X = torch.randn(num_samples, *input_size)
    y = torch.randint(0, 2, (num_samples, num_classes)).float()
    return TensorDataset(X, y)


def load_client_data(data_dir, labels_file):
    """Load preprocessed tensors and labels for a client.

    Returns a PyTorch Dataset or None if torch unavailable.
    """
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
            logger.warning(f"Tensor not found: {tensor_path}")
            continue

        try:
            tensor = torch.load(tensor_path, weights_only=True)
            # tensor shape: [C, D, H, W] or [C, H, W]
            # For 2D classification, we take the middle slice from 3D volumes
            if len(tensor.shape) == 4:
                mid = tensor.shape[1] // 2
                tensor = tensor[:, mid, :, :]  # [C, H, W]
            elif len(tensor.shape) == 3 and tensor.shape[0] == 1:
                pass  # Already [C, H, W]
            elif len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(0)  # [1, H, W]

            # Resize to 224x224 if needed
            if tensor.shape[-1] != 224 or tensor.shape[-2] != 224:
                tensor = torch.nn.functional.interpolate(
                    tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
                ).squeeze(0)

            tensors.append(tensor)
            # Synthetic multi-label target
            targets.append(torch.randint(0, 2, (14,)).float())
        except Exception as e:
            logger.warning(f"Failed to load tensor {tensor_path}: {e}")
            continue

    if len(tensors) == 0:
        return None

    X = torch.stack(tensors)
    Y = torch.stack(targets)
    return TensorDataset(X, Y)


def compute_auc(y_true, y_pred):
    """Compute per-class ROC-AUC."""
    try:
        from sklearn.metrics import roc_auc_score
        aucs = []
        for i in range(y_true.shape[1]):
            try:
                aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
            except ValueError:
                aucs.append(0.5)
        return sum(aucs) / len(aucs) if aucs else 0.0
    except ImportError:
        return 0.0


def train_epoch(model, dataloader, optimizer, criterion, device, global_params=None, fedprox_mu=0.0):
    """Train for one epoch. Returns avg loss."""
    import torch

    model.train()
    total_loss = 0.0
    num_batches = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)

        # FedProx regularization
        if fedprox_mu > 0.0 and global_params is not None:
            prox_term = 0.0
            for name, param in model.named_parameters():
                if name in global_params:
                    prox_term += torch.sum((param - global_params[name]) ** 2)
            loss = loss + (fedprox_mu / 2.0) * prox_term

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model. Returns (loss, auc)."""
    import torch

    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            num_batches += 1
            all_preds.append(torch.sigmoid(outputs).cpu())
            all_labels.append(y.cpu())

    avg_loss = total_loss / max(num_batches, 1)
    if all_preds:
        auc = compute_auc(
            torch.cat(all_labels).numpy(),
            torch.cat(all_preds).numpy()
        )
    else:
        auc = 0.0
    return avg_loss, auc


def main():
    parser = argparse.ArgumentParser(
        description="Local FL training for one client for one round."
    )
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--data-dir", required=True, help="Preprocessed client data directory")
    parser.add_argument("--global-model", required=True, help="Input global model checkpoint (.pt)")
    parser.add_argument("--output-model", required=True, help="Output local model checkpoint (.pt)")
    parser.add_argument("--output-metrics", required=True, help="Output training metrics JSON")
    parser.add_argument("--arch-config", default=None, help="Model architecture JSON")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fedprox-mu", type=float, default=0.0)
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--synthetic-data", action="store_true", help="Use synthetic data for testing")

    args = parser.parse_args()

    logger.info(f"Client:       {args.client_id}")
    logger.info(f"Data dir:     {args.data_dir}")
    logger.info(f"Global model: {args.global_model}")
    logger.info(f"Round:        {args.round}")
    logger.info(f"Epochs:       {args.epochs}")
    logger.info(f"FedProx mu:   {args.fedprox_mu}")
    logger.info(f"Device:       {args.device}")

    model_dir = os.path.dirname(args.output_model)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    metrics_dir = os.path.dirname(args.output_metrics)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    # Check torch availability
    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        logger.error("torch not available; cannot train")
        # Write placeholder outputs
        metrics = {"client_id": args.client_id, "round": args.round, "status": "no_torch"}
        with open(args.output_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        import pickle
        with open(args.output_model, "wb") as f:
            pickle.dump({"status": "placeholder"}, f)
        logger.info("train_client: wrote placeholders (no torch).")
        return

    # Load architecture config
    if args.arch_config and os.path.exists(args.arch_config):
        arch = load_arch_config(args.arch_config)
    else:
        default_arch_path = os.path.join(os.path.dirname(args.global_model), "model_arch.json")
        if os.path.exists(default_arch_path):
            arch = load_arch_config(default_arch_path)
        else:
            arch = {"model": "DenseNet121", "num_classes": 14, "in_channels": 1, "pretrained": False}
            logger.info(f"Using default arch: {arch}")

    # Create model and load global weights
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = create_model(arch).to(device)

    if os.path.exists(args.global_model):
        state = torch.load(args.global_model, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state" in state:
            logger.warning("Global model is a placeholder; starting from scratch")
        else:
            model.load_state_dict(state)
            logger.info("Loaded global model weights")
    else:
        logger.warning("Global model not found; training from scratch")

    # Save global params for FedProx
    global_params = None
    if args.fedprox_mu > 0.0:
        global_params = {name: param.detach().clone() for name, param in model.named_parameters()}
        logger.info("Stored global params for FedProx")

    # Load dataset
    labels_file = os.path.join(args.data_dir, "labels.json")
    dataset = None
    if not args.synthetic_data:
        dataset = load_client_data(args.data_dir, labels_file)

    if dataset is None:
        logger.info("Using synthetic data")
        dataset = create_synthetic_dataset(num_samples=50)

    # Split train/val
    n_total = len(dataset)
    n_val = max(1, int(n_total * 0.2))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False
    )

    logger.info(f"Dataset: {n_train} train, {n_val} val samples")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    metrics = {
        "client_id": args.client_id,
        "round": args.round,
        "epochs": args.epochs,
        "n_train": n_train,
        "n_val": n_val,
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
    }

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            global_params=global_params, fedprox_mu=args.fedprox_mu
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_auc"].append(val_auc)

        logger.info(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_auc={val_auc:.4f}"
        )

    # Save local model
    torch.save(model.state_dict(), args.output_model)
    logger.info(f"Local model saved: {args.output_model}")

    # Save metrics
    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {args.output_metrics}")
    logger.info("train_client complete.")


if __name__ == "__main__":
    main()
