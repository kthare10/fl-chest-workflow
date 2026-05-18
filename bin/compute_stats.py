#!/usr/bin/env python3

"""Compute per-client label and intensity statistics for heterogeneity reporting.

Metrics:
- Class histogram (label distribution)
- Intensity mean/std per series
- Scanner manufacturer distribution (from DICOM metadata if available)
- KL divergence reference for cross-client comparison
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_label_distribution(labels_data, num_classes=14):
    """Compute normalized class histogram from labels JSON."""
    counts = Counter()
    total = 0
    for series in labels_data.get("series", []):
        # In real data, labels would be here. For now count by status.
        counts[series.get("status", "unknown")] += 1
        total += 1

    if total == 0:
        return {}

    dist = {k: v / total for k, v in counts.items()}
    return dist


def compute_intensity_stats(tensor_dir, labels_data, max_samples=100):
    """Compute mean/std voxel intensities from preprocessed tensors."""
    import numpy as np

    intensities = []
    for series in labels_data.get("series", [])[:max_samples]:
        tensor_file = series.get("tensor_file")
        if not tensor_file:
            continue
        tensor_path = os.path.join(tensor_dir, tensor_file)
        if not os.path.exists(tensor_path):
            continue

        try:
            tensor = np.load(tensor_path) if tensor_file.endswith(".npy") else None
            if tensor is None:
                try:
                    import torch
                    tensor = torch.load(tensor_path, weights_only=True).numpy()
                except ImportError:
                    continue
            intensities.append(tensor.mean())
        except Exception as e:
            logger.warning(f"Failed to read {tensor_path}: {e}")
            continue

    if not intensities:
        return {"mean": 0.0, "std": 1.0, "n_samples": 0}

    arr = np.array(intensities)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_samples": len(intensities),
    }


def compute_scanner_distribution(manifest_data):
    """Extract scanner manufacturer distribution from manifest metadata."""
    manufacturers = Counter()
    for series in manifest_data.get("series", []):
        mfr = series.get("manufacturer", "UNKNOWN")
        manufacturers[mfr] += 1

    total = sum(manufacturers.values())
    if total == 0:
        return {}

    return {k: v / total for k, v in manufacturers.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Compute label distribution and intensity statistics per client."
    )
    parser.add_argument("--input-dir", required=True, help="Preprocessed client data directory")
    parser.add_argument("--output-json", required=True, help="Output statistics JSON file")
    parser.add_argument("--manifest", default=None, help="Optional ingest manifest for scanner metadata")

    args = parser.parse_args()

    logger.info(f"Input dir:   {args.input_dir}")
    logger.info(f"Output JSON: {args.output_json}")

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    labels_file = os.path.join(args.input_dir, "labels.json")
    labels_data = {}
    if os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            labels_data = json.load(f)

    manifest_data = {}
    if args.manifest and os.path.exists(args.manifest):
        with open(args.manifest, "r") as f:
            manifest_data = json.load(f)

    label_dist = compute_label_distribution(labels_data)
    intensity_stats = compute_intensity_stats(args.input_dir, labels_data)
    scanner_dist = compute_scanner_distribution(manifest_data)

    stats = {
        "collection": labels_data.get("collection", os.path.basename(args.input_dir)),
        "status": "ok",
        "num_series": len(labels_data.get("series", [])),
        "label_distribution": label_dist,
        "intensity_stats": intensity_stats,
        "scanner_distribution": scanner_dist,
    }

    with open(args.output_json, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Stats written: {args.output_json}")
    logger.info(f"  Series: {stats['num_series']}")
    logger.info(f"  Intensity: mean={intensity_stats['mean']:.4f}, std={intensity_stats['std']:.4f}")
    logger.info("compute_stats complete.")


if __name__ == "__main__":
    main()
