#!/usr/bin/env python3

"""Preprocess NIH Chest X-Ray14 images into standardized PyTorch tensors.

Reads:
- PNG images from a directory (e.g., images/)
- Data_Entry_2017.csv with columns: Image Index, Finding Labels, ...

Outputs:
- Grayscale 224x224 .pt tensors
- labels.json compatible with evaluate.py

Supports --synthetic for testing.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# NIH 14 pathology labels in standard order
NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]


def parse_nih_labels(finding_labels_str):
    """Convert NIH 'Finding Labels' string to binary vector."""
    findings = [f.strip() for f in str(finding_labels_str).split("|")]
    vector = [1.0 if label in findings else 0.0 for label in NIH_LABELS]
    return vector


def load_image(image_path):
    """Load image as grayscale numpy array (H, W)."""
    try:
        from PIL import Image
        img = Image.open(image_path).convert("L")
        return img
    except ImportError:
        logger.warning("PIL not available; cannot load real images")
        return None
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return None


def preprocess_image(image_path, output_dir, series_uid):
    """Load image, resize to 224x224, normalize, save as .pt tensor."""
    import numpy as np

    img = load_image(image_path)
    if img is None:
        return None

    try:
        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Converts to [0,1] range, tensor shape (1, H, W)
        ])
        tensor = transform(img)

        out_path = os.path.join(output_dir, f"{series_uid}.pt")
        torch.save(tensor, out_path)

        return {
            "tensor_file": os.path.basename(out_path),
            "tensor_shape": list(tensor.shape),
        }
    except ImportError:
        # Numpy fallback
        arr = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
        out_path = os.path.join(output_dir, f"{series_uid}.npy")
        np.save(out_path, arr)
        return {
            "tensor_file": os.path.basename(out_path),
            "tensor_shape": list(arr.shape),
        }


def generate_synthetic_nih(output_dir, labels_file, num_samples=50):
    """Generate synthetic NIH-style preprocessed tensors for testing."""
    import numpy as np

    logger.info("Generating synthetic NIH preprocessed data")
    os.makedirs(output_dir, exist_ok=True)

    labels = {
        "collection": "NIH-Chest-XRay14",
        "status": "synthetic",
        "num_series": num_samples,
        "series": [],
    }

    for i in range(num_samples):
        series_uid = f"NIH_SYNTH_{i:05d}"
        y_vec = parse_nih_labels("Infiltration|Nodule")

        try:
            import torch
            tensor = torch.rand(1, 224, 224)
            out_path = os.path.join(output_dir, f"{series_uid}.pt")
            torch.save(tensor, out_path)
            tensor_file = os.path.basename(out_path)
            tensor_shape = list(tensor.shape)
        except ImportError:
            arr = np.random.rand(224, 224).astype(np.float32)
            out_path = os.path.join(output_dir, f"{series_uid}.npy")
            np.save(out_path, arr)
            tensor_file = os.path.basename(out_path)
            tensor_shape = list(arr.shape)

        labels["series"].append({
            "SeriesInstanceUID": series_uid,
            "tensor_file": tensor_file,
            "tensor_shape": tensor_shape,
            "labels": y_vec,
            "patient_id": f"SYNTH_P{i:05d}",
            "status": "synthetic",
        })

    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=2)

    logger.info(f"Synthetic NIH preprocessed data written to {output_dir}")
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess NIH Chest X-Ray14 images into standardized PyTorch tensors."
    )
    parser.add_argument("--images-dir", required=True, help="Directory with NIH PNG images")
    parser.add_argument("--labels-csv", required=True, help="Path to Data_Entry_2017.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for preprocessed .pt tensors")
    parser.add_argument("--output-labels", required=True, help="Output labels JSON")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples (for testing)")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data for testing")

    args = parser.parse_args()

    logger.info(f"Images dir:     {args.images_dir}")
    logger.info(f"Labels CSV:     {args.labels_csv}")
    logger.info(f"Output dir:     {args.output_dir}")
    logger.info(f"Output labels:  {args.output_labels}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.synthetic:
        generate_synthetic_nih(args.output_dir, args.output_labels, num_samples=args.max_samples or 50)
        logger.info("Synthetic NIH preprocessing complete.")
        return

    # --- Real preprocessing ---
    if not os.path.exists(args.labels_csv):
        logger.error(f"Labels CSV not found: {args.labels_csv}")
        sys.exit(1)

    df = pd.read_csv(args.labels_csv)
    logger.info(f"Loaded {len(df)} rows from {args.labels_csv}")

    if args.max_samples:
        df = df.head(args.max_samples)
        logger.info(f"Limited to {len(df)} samples")

    labels = {
        "collection": "NIH-Chest-XRay14",
        "status": "preprocessed",
        "num_series": 0,
        "series": [],
    }

    skipped = 0
    for idx, row in df.iterrows():
        image_name = row.get("Image Index", "")
        if not image_name:
            skipped += 1
            continue

        image_path = os.path.join(args.images_dir, image_name)
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            skipped += 1
            continue

        series_uid = Path(image_name).stem
        result = preprocess_image(image_path, args.output_dir, series_uid)
        if result is None:
            skipped += 1
            continue

        y_vec = parse_nih_labels(row.get("Finding Labels", ""))
        labels["series"].append({
            "SeriesInstanceUID": series_uid,
            **result,
            "labels": y_vec,
            "patient_id": str(row.get("Patient ID", "unknown")),
        })
        labels["num_series"] += 1

    with open(args.output_labels, "w") as f:
        json.dump(labels, f, indent=2)

    logger.info(
        f"NIH preprocessing complete: {labels['num_series']} series processed, "
        f"{skipped} skipped"
    )


if __name__ == "__main__":
    main()
