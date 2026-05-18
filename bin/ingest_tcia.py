#!/usr/bin/env python3

"""Download and validate DICOM data for one TCIA collection via NBIA REST API.

Supports:
- Live download from TCIA NBIA API (default)
- Synthetic data generation for testing (--synthetic)
- Manifest-based selective download (--manifest-csv)

References:
- TCIA NBIA API: https://wiki.cancerimagingarchive.net/x/fQATBQ
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

NBIA_BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def _create_session(verify_ssl=True):
    """Create a requests.Session with retry adapter."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,  # 1s, 2s, 4s, 8s, 16s
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    if not verify_ssl:
        session.verify = False
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return session


def _fetch_json(url, params=None, retries=3, timeout=60, session=None, verify_ssl=True):
    """Fetch JSON from NBIA API with retries and SSL resilience."""
    if session is None:
        session = _create_session(verify_ssl=verify_ssl)

    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt * 2, 30))  # max 30s backoff
            else:
                raise


def fetch_series_for_collection(collection_name, verify_ssl=True):
    """Return list of series metadata dicts for a TCIA collection."""
    url = f"{NBIA_BASE_URL}/getSeries"
    logger.info(f"Querying NBIA for collection: {collection_name}")
    session = _create_session(verify_ssl=verify_ssl)
    data = _fetch_json(url, params={"Collection": collection_name}, session=session, verify_ssl=verify_ssl)
    if not isinstance(data, list):
        logger.error(f"Unexpected response format from NBIA: {type(data)}")
        sys.exit(1)
    logger.info(f"Found {len(data)} series in collection {collection_name}")
    return data


def download_series(series_uid, output_dir, validate=True, session=None):
    """Download one DICOM series as a zip and extract to output_dir/series_uid/."""
    if session is None:
        session = _create_session()

    url = f"{NBIA_BASE_URL}/getImage"
    series_dir = os.path.join(output_dir, series_uid)
    os.makedirs(series_dir, exist_ok=True)

    zip_path = os.path.join(output_dir, f"{series_uid}.zip")

    logger.info(f"Downloading series {series_uid} ...")
    resp = session.get(url, params={"SeriesInstanceUID": series_uid}, timeout=300)
    resp.raise_for_status()

    with open(zip_path, "wb") as f:
        f.write(resp.content)

    # Extract
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(series_dir)

    os.remove(zip_path)

    # Basic validation
    dicom_files = list(Path(series_dir).rglob("*.dcm"))
    if len(dicom_files) == 0:
        logger.warning(f"No .dcm files found in {series_dir}")
        return None

    logger.info(f"  -> {len(dicom_files)} DICOM files extracted")
    return len(dicom_files)


def validate_dicom_headers(dicom_path):
    """Validate a single DICOM file and return metadata dict."""
    try:
        import pydicom
    except ImportError:
        logger.warning("pydicom not available; skipping header validation")
        return {"valid": True, "modality": "UNKNOWN"}

    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        modality = getattr(ds, "Modality", "UNKNOWN")
        manufacturer = getattr(ds, "Manufacturer", "UNKNOWN")
        slice_thickness = getattr(ds, "SliceThickness", None)
        return {
            "valid": True,
            "modality": modality,
            "manufacturer": manufacturer,
            "slice_thickness": float(slice_thickness) if slice_thickness else None,
        }
    except Exception as e:
        logger.warning(f"Failed to read DICOM {dicom_path}: {e}")
        return {"valid": False, "error": str(e)}


def generate_synthetic_data(collection_name, output_dir, num_studies=3, num_slices=10):
    """Generate synthetic DICOM-like files for testing without network."""
    logger.info(f"Generating synthetic data for {collection_name}: {num_studies} studies, {num_slices} slices each")

    manifest = {
        "collection": collection_name,
        "status": "synthetic",
        "num_studies": num_studies,
        "series": [],
    }

    for s in range(num_studies):
        series_uid = f"1.2.3.4.5.{collection_name.replace(' ', '_')}.{s}"
        series_dir = os.path.join(output_dir, series_uid)
        os.makedirs(series_dir, exist_ok=True)

        # Try to write minimal DICOM; fall back to binary placeholder
        for z in range(num_slices):
            dcm_path = os.path.join(series_dir, f"slice_{z:03d}.dcm")
            try:
                import pydicom
                from pydicom.dataset import FileDataset, FileMetaDataset
                from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian

                file_meta = FileMetaDataset()
                file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
                file_meta.MediaStorageSOPInstanceUID = series_uid
                file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

                ds = FileDataset(dcm_path, {}, file_meta=file_meta, preamble=b"\x00" * 128)
                ds.PatientName = f"SyntheticPatient{s}"
                ds.PatientID = f"SYN{s:03d}"
                ds.Modality = "CT"
                ds.SeriesInstanceUID = series_uid
                ds.SliceThickness = 1.0
                ds.Manufacturer = "SyntheticScanner"
                ds.Rows = 128
                ds.Columns = 128
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.PixelRepresentation = 1
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                import numpy as np
                ds.PixelData = np.random.randint(-1000, 1000, size=(128, 128), dtype=np.int16).tobytes()
                ds.save_as(dcm_path)
            except ImportError:
                # Pure placeholder
                with open(dcm_path, "wb") as f:
                    f.write(b"\x00" * 1024)

        manifest["series"].append({
            "SeriesInstanceUID": series_uid,
            "num_slices": num_slices,
            "modality": "CT",
        })

    # Note: manifest is written by the caller (main()) to args.output_manifest,
    # not here, so that the declared Pegasus LFN matches the actual path.
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Ingest TCIA data: validate DICOMs and emit manifest.json for downstream pipeline."
    )
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--output-dir", required=True,
                        help="Output directory where validated data is staged (usually a symlink or copy)")
    parser.add_argument("--output-manifest", required=True,
                        help="Output manifest JSON (declared Pegasus File)")
    parser.add_argument("--input-tar", default=None,
                        help="Path to PRE-STAGED tar.gz (from download_tcia.py). Extracted in job working dir.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic test data instead of using real data")
    parser.add_argument("--synthetic-fallback", action="store_true",
                        help="If no pre-staged data found, fall back to synthetic")
    parser.add_argument("--max-series", type=int, default=None,
                        help="Cap number of series to include in manifest")
    parser.add_argument("--validate-headers", action="store_true", default=True,
                        help="Validate DICOM headers")

    # Legacy network flags (kept for compatibility but ignored when --input-dir is used)
    parser.add_argument("--manifest-csv", default=None, help="[Legacy] CSV with SeriesInstanceUIDs — use download_tcia.py instead")
    parser.add_argument("--no-verify-ssl", action="store_true", help="[Legacy] Disable SSL verification — use download_tcia.py instead")

    args = parser.parse_args()

    logger.info(f"Collection:      {args.collection_name}")
    logger.info(f"Output dir:      {args.output_dir}")
    logger.info(f"Output manifest: {args.output_manifest}")
    logger.info(f"Input tar:       {args.input_tar}")
    logger.info(f"Synthetic mode:  {args.synthetic}")
    logger.info(f"Synthetic fallback: {args.synthetic_fallback}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Synthetic path ---
    if args.synthetic:
        manifest = generate_synthetic_data(args.collection_name, args.output_dir)
        with open(args.output_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Synthetic generation complete.")
        return

    # --- Pre-staged tar path ---
    if args.input_tar and os.path.exists(args.input_tar):
        logger.info(f"Using pre-staged tar: {args.input_tar}")

        import tarfile
        extract_dir = os.path.join(args.output_dir, "raw")
        os.makedirs(extract_dir, exist_ok=True)

        with tarfile.open(args.input_tar, "r:gz") as tar:
            tar.extractall(path=extract_dir)

        # The tar extracts to extract_dir/<collection_name>/
        input_collection_dir = os.path.join(extract_dir, args.collection_name)
        if not os.path.isdir(input_collection_dir):
            # Fallback: maybe the tar contained flat directories
            candidates = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
            if candidates:
                input_collection_dir = os.path.join(extract_dir, candidates[0])
            else:
                logger.error(f"Could not find collection directory inside tar")
                if args.synthetic_fallback:
                    logger.info("Falling back to synthetic data")
                    manifest = generate_synthetic_data(args.collection_name, args.output_dir)
                    with open(args.output_manifest, "w") as f:
                        json.dump(manifest, f, indent=2)
                    return
                else:
                    sys.exit(1)

        # If a manifest already exists in the extracted dir, reuse it (fast path)
        prestaged_manifest_path = os.path.join(input_collection_dir, "manifest.json")
        if os.path.exists(prestaged_manifest_path):
            logger.info(f"Reusing pre-staged manifest: {prestaged_manifest_path}")
            with open(prestaged_manifest_path, "r") as f:
                manifest = json.load(f)
            if args.validate_headers:
                for meta in manifest.get("series", []):
                    uid = meta.get("SeriesInstanceUID")
                    if uid:
                        series_dir = os.path.join(input_collection_dir, uid)
                        dcm_files = list(Path(series_dir).glob("*.dcm"))
                        if dcm_files:
                            hdr = validate_dicom_headers(dcm_files[0])
                            meta.update(hdr)
            manifest["status"] = "prestaged"
            with open(args.output_manifest, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Ingest complete from pre-staged manifest: {len(manifest['series'])} series")
            return

        # Otherwise scan the extracted directory and build manifest
        manifest = {
            "collection": args.collection_name,
            "status": "prestaged",
            "num_series_total": 0,
            "series": [],
        }

        series_dirs = [d for d in os.listdir(input_collection_dir)
                       if os.path.isdir(os.path.join(input_collection_dir, d))]

        if args.max_series and len(series_dirs) > args.max_series:
            series_dirs = series_dirs[:args.max_series]

        for series_uid in series_dirs:
            src = os.path.join(input_collection_dir, series_uid)
            dst = os.path.join(args.output_dir, series_uid)

            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    import shutil
                    shutil.copytree(src, dst)

            dcm_files = list(Path(dst).rglob("*.dcm"))
            meta = {
                "SeriesInstanceUID": series_uid,
                "num_files": len(dcm_files),
            }
            if args.validate_headers and dcm_files:
                meta.update(validate_dicom_headers(dcm_files[0]))
            manifest["series"].append(meta)

        manifest["num_series_total"] = len(manifest["series"])

        with open(args.output_manifest, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Ingest complete from pre-staged tar: {len(manifest['series'])} series")
        return

    # --- Fallback: synthetic if no input-tar and synthetic-fallback is on ---
    if args.synthetic_fallback:
        logger.info("No pre-staged tar available; falling back to synthetic")
        manifest = generate_synthetic_data(args.collection_name, args.output_dir)
        with open(args.output_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        return

    # --- Legacy live download path (should rarely be used from inside workflow) ---
    logger.error(
        "No pre-staged tar provided (--input-tar). "
        "Run download_tcia.py on the submit host first, then use --input-tar."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
