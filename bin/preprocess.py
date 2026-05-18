#!/usr/bin/env python3

"""Preprocess DICOM volumes into standardized PyTorch tensors using MONAI.

Pipeline per DICOM series:
1. Load DICOM series (MONAI LoadImage or SimpleITK)
2. Resample to target spacing (MONAI Spacingd)
3. Apply intensity windowing (MONAI ScaleIntensityRanged)
4. Save as .pt tensor
5. Generate labels.json with metadata

Supports --synthetic for testing without actual DICOM libraries.
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


def load_dicom_series(series_dir):
    """Load a DICOM series directory into a numpy array and metadata."""
    try:
        import SimpleITK as sitk
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(series_dir)
        if not dicom_names:
            logger.warning(f"No DICOM series found in {series_dir}")
            return None, None
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        arr = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        meta = {
            "spacing": list(spacing),
            "origin": list(origin),
            "direction": list(direction),
            "shape": list(arr.shape),
        }
        return arr, meta
    except ImportError:
        logger.warning("SimpleITK not available; cannot load real DICOMs")
        return None, None
    except Exception as e:
        logger.warning(f"Failed to load DICOM from {series_dir}: {e}")
        return None, None


def resample_volume(arr, original_spacing, target_spacing=1.0):
    """Resample a 3D volume to target isotropic spacing using scipy."""
    import numpy as np
    from scipy.ndimage import zoom

    if arr is None or original_spacing is None:
        return arr, None

    zoom_factors = [orig / target_spacing for orig in original_spacing]
    resampled = zoom(arr, zoom=zoom_factors, order=1)
    new_shape = list(resampled.shape)
    return resampled, new_shape


def apply_window(arr, center=40, width=400):
    """Apply CT windowing and scale to [0, 1]."""
    import numpy as np

    if arr is None:
        return None

    min_val = center - width / 2
    max_val = center + width / 2
    windowed = np.clip(arr, min_val, max_val)
    windowed = (windowed - min_val) / (max_val - min_val)
    return windowed.astype(np.float32)


def preprocess_series(series_dir, output_dir, spacing, window_center, window_width):
    """Process one DICOM series and write .pt tensor. Returns metadata dict."""
    import numpy as np

    series_uid = os.path.basename(series_dir)
    logger.info(f"Processing series {series_uid} ...")

    arr, meta = load_dicom_series(series_dir)
    if arr is None:
        return None

    # Resample
    if len(meta.get("spacing", [])) >= 3:
        original_spacing = meta["spacing"]
    else:
        original_spacing = [1.0, 1.0, 1.0]

    arr, new_shape = resample_volume(arr, original_spacing, target_spacing=spacing)
    meta["resampled_shape"] = new_shape
    meta["target_spacing"] = spacing

    # Window and normalize
    arr = apply_window(arr, center=window_center, width=window_width)
    if arr is None:
        return None

    # Save as PyTorch tensor
    try:
        import torch
        tensor = torch.from_numpy(arr).unsqueeze(0)  # Add channel dimension
        out_path = os.path.join(output_dir, f"{series_uid}.pt")
        torch.save(tensor, out_path)
        meta["tensor_file"] = os.path.basename(out_path)
        meta["tensor_shape"] = list(tensor.shape)
        logger.info(f"  -> Saved {out_path} shape={tensor.shape}")
    except ImportError:
        # Fallback: save numpy
        out_path = os.path.join(output_dir, f"{series_uid}.npy")
        np.save(out_path, arr)
        meta["tensor_file"] = os.path.basename(out_path)
        meta["tensor_shape"] = list(arr.shape)
        logger.info(f"  -> Saved {out_path} shape={arr.shape} (numpy fallback)")

    return meta


def generate_synthetic_preprocessed(output_dir, labels_file, num_series=3):
    """Generate synthetic preprocessed tensors for testing."""
    import numpy as np

    logger.info("Generating synthetic preprocessed data")
    os.makedirs(output_dir, exist_ok=True)

    labels = {
        "status": "synthetic",
        "num_series": num_series,
        "series": [],
    }

    for i in range(num_series):
        series_uid = f"1.2.3.4.5.SYNTHETIC.{i}"
        arr = np.random.rand(10, 128, 128).astype(np.float32)

        try:
            import torch
            tensor = torch.from_numpy(arr).unsqueeze(0)
            out_path = os.path.join(output_dir, f"{series_uid}.pt")
            torch.save(tensor, out_path)
            tensor_file = os.path.basename(out_path)
            tensor_shape = list(tensor.shape)
        except ImportError:
            out_path = os.path.join(output_dir, f"{series_uid}.npy")
            np.save(out_path, arr)
            tensor_file = os.path.basename(out_path)
            tensor_shape = list(arr.shape)

        labels["series"].append({
            "SeriesInstanceUID": series_uid,
            "tensor_file": tensor_file,
            "tensor_shape": tensor_shape,
            "status": "synthetic",
        })

    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=2)

    logger.info(f"Synthetic preprocessed data written to {output_dir}")
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DICOM volumes into standardized PyTorch tensors."
    )
    parser.add_argument("--input-dir", required=True, help="Directory with raw DICOM data")
    parser.add_argument("--input-tar", default=None, help="Optional pre-staged tar.gz; extracted into input-dir before processing")
    parser.add_argument("--output-dir", required=True, help="Directory for preprocessed .pt tensors")
    parser.add_argument("--output-labels", required=True, help="Output labels JSON (declared Pegasus File)")
    parser.add_argument("--manifest", default=None, help="Path to manifest.json (default: input-dir/manifest.json)")
    parser.add_argument("--spacing", type=float, default=1.0, help="Target isotropic spacing in mm")
    parser.add_argument("--window-center", type=int, default=40, help="Window center (HU) for CT")
    parser.add_argument("--window-width", type=int, default=400, help="Window width (HU) for CT")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data for testing")

    args = parser.parse_args()

    logger.info(f"Input dir:      {args.input_dir}")
    logger.info(f"Input tar:      {args.input_tar}")
    logger.info(f"Output dir:     {args.output_dir}")
    logger.info(f"Output labels:  {args.output_labels}")
    logger.info(f"Spacing:        {args.spacing} mm")
    logger.info(f"Window:         C={args.window_center} W={args.window_width}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Extract pre-staged tar if provided ---
    if args.input_tar and os.path.exists(args.input_tar):
        import tarfile
        os.makedirs(args.input_dir, exist_ok=True)
        logger.info(f"Extracting tar: {args.input_tar}")
        with tarfile.open(args.input_tar, "r:gz") as tar:
            tar.extractall(path=args.input_dir)
        logger.info(f"Extraction complete into {args.input_dir}")

    if args.synthetic:
        labels = generate_synthetic_preprocessed(args.output_dir, args.output_labels)
        logger.info("Synthetic preprocessing complete.")
        return

    # --- Real preprocessing ---
    manifest_path = args.manifest or os.path.join(args.input_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    labels = {
        "collection": manifest.get("collection", "unknown"),
        "status": "preprocessed",
        "num_series": 0,
        "series": [],
    }

    skipped = 0
    for series_meta in manifest.get("series", []):
        uid = series_meta.get("SeriesInstanceUID")
        if not uid:
            continue

        series_dir = os.path.join(args.input_dir, uid)
        if not os.path.isdir(series_dir):
            logger.warning(f"Series directory not found: {series_dir}")
            skipped += 1
            continue

        result = preprocess_series(
            series_dir,
            args.output_dir,
            spacing=args.spacing,
            window_center=args.window_center,
            window_width=args.window_width,
        )
        if result:
            labels["series"].append({
                "SeriesInstanceUID": uid,
                **result,
            })
            labels["num_series"] += 1
        else:
            skipped += 1

    # Write labels to declared Pegasus output path
    with open(args.output_labels, "w") as f:
        json.dump(labels, f, indent=2)

    logger.info(
        f"Preprocessing complete: {labels['num_series']} series processed, "
        f"{skipped} skipped"
    )


if __name__ == "__main__":
    main()
