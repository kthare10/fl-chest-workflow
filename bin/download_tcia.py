#!/usr/bin/env python3

"""Standalone TCIA data downloader — run on submit host BEFORE submitting workflow.

Downloads DICOM series, packages into per-collection tar.gz archives.
These archives become Pegasus inputs and travel with jobs via Condor I/O.

Usage:
    ./download_tcia.py --collections LIDC-IDRI NSCLC-Radiomics --output-dir data/
    # Produces data/LIDC-IDRI.tar.gz, data/NSCLC-Radiomics.tar.gz, and data/download_manifest.json
"""

import argparse
import csv
import json
import logging
import os
import sys
import tarfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

NBIA_BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def _create_session(verify_ssl=True):
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
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
                time.sleep(min(2 ** attempt * 2, 30))
            else:
                raise


def fetch_series_for_collection(collection_name, verify_ssl=True):
    url = f"{NBIA_BASE_URL}/getSeries"
    logger.info(f"Querying NBIA for collection: {collection_name}")
    session = _create_session(verify_ssl=verify_ssl)
    data = _fetch_json(url, params={"Collection": collection_name}, session=session, verify_ssl=verify_ssl)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected response format from NBIA: {type(data)}")
    logger.info(f"Found {len(data)} series in collection {collection_name}")
    return data


def download_series(series_uid, output_dir, session=None):
    if session is None:
        session = _create_session()

    url = f"{NBIA_BASE_URL}/getImage"
    series_dir = os.path.join(output_dir, series_uid)

    # Skip if already downloaded
    if os.path.isdir(series_dir) and any(Path(series_dir).rglob("*.dcm")):
        dcm_files = list(Path(series_dir).rglob("*.dcm"))
        return len(dcm_files)

    os.makedirs(series_dir, exist_ok=True)
    import zipfile
    zip_path = os.path.join(output_dir, f"{series_uid}.zip")

    logger.info(f"Downloading series {series_uid} ...")
    resp = session.get(url, params={"SeriesInstanceUID": series_uid}, timeout=300)
    resp.raise_for_status()

    with open(zip_path, "wb") as f:
        f.write(resp.content)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(series_dir)

    os.remove(zip_path)

    dicom_files = list(Path(series_dir).rglob("*.dcm"))
    return len(dicom_files)


def package_collection(collection_name, collection_dir, output_dir, session=None):
    """Download all series for a collection and tar.gz them."""
    verify_ssl = True
    series_list = []
    try:
        series_list = fetch_series_for_collection(collection_name, verify_ssl=verify_ssl)
    except Exception as e:
        logger.error(f"Failed to query NBIA for {collection_name}: {e}")
        return None

    os.makedirs(collection_dir, exist_ok=True)
    if session is None:
        session = _create_session()

    manifest = {
        "collection": collection_name,
        "status": "live",
        "num_series_total": len(series_list),
        "series": [],
    }

    failed = 0
    total_series = len(series_list)
    for idx, s in enumerate(series_list):
        uid = s.get("SeriesInstanceUID")
        if not uid:
            continue
        try:
            count = download_series(uid, collection_dir, session=session)
            manifest["series"].append({"SeriesInstanceUID": uid, "num_files": count})
        except Exception as e:
            logger.error(f"Failed to download series {uid}: {e}")
            failed += 1

        # Progress logging every 100 series or at the end
        if (idx + 1) % 100 == 0 or (idx + 1) == total_series:
            logger.info(
                f"Progress [{collection_name}]: {idx + 1}/{total_series} series "
                f"({failed} failed, {len(manifest['series'])} successful)"
            )

    # Write per-collection manifest inside directory
    with open(os.path.join(collection_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Create tar.gz
    tar_path = os.path.join(output_dir, f"{collection_name}.tar.gz")
    logger.info(f"Creating archive: {tar_path}")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(collection_dir, arcname=collection_name)

    logger.info(
        f"Collection {collection_name}: {len(manifest['series'])} series, {failed} failures -> {tar_path}"
    )

    return tar_path


def main():
    parser = argparse.ArgumentParser(description="Pre-stage TCIA DICOM data as tar.gz archives for Pegasus")
    parser.add_argument("--collections", nargs="+", required=True, help="TCIA collection names")
    parser.add_argument("--output-dir", required=True, help="Directory for tar.gz output")
    parser.add_argument("--max-series", type=int, default=None, help="Cap series per collection")
    parser.add_argument("--no-verify-ssl", action="store_true", help="Disable SSL verification")
    parser.add_argument("--resume", action="store_true", help="Skip collections where tar.gz already exists")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    session = _create_session(verify_ssl=not args.no_verify_ssl)

    # Load existing summary if resuming
    summary_path = os.path.join(args.output_dir, "download_manifest.json")
    summary = {}
    if args.resume and os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        logger.info(f"Loaded existing summary: {summary_path}")

    for collection in args.collections:
        tar_path = os.path.join(args.output_dir, f"{collection}.tar.gz")

        # Skip if tar.gz already exists and --resume is set
        if args.resume and os.path.exists(tar_path):
            logger.info(f"Collection {collection}: tar.gz already exists, skipping (--resume)")
            if collection not in summary:
                summary[collection] = {"tar": os.path.abspath(tar_path), "status": "skipped_existing"}
            continue

        # Download to a temp dir inside output, then tar
        temp_dir = os.path.join(args.output_dir, ".tmp", collection)
        result = package_collection(collection, temp_dir, args.output_dir, session=session)
        if result:
            summary[collection] = {"tar": os.path.abspath(result), "status": "ok"}
        else:
            summary[collection] = {"status": "failed"}

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Pre-staging complete. Summary: {summary_path}")
    logger.info("Next: submit workflow with --tcia-data-dir pointing to this directory.")


if __name__ == "__main__":
    main()
