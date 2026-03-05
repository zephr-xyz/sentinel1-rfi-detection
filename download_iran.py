#!/usr/bin/env python3
"""
Download selected Iran-wide Sentinel-1 scenes and run the RFI detection + map pipeline.

Reads product selections from output/iran_selected_products.json,
downloads each via CDSE, then runs create_map.py to produce the interactive map.
"""

import json
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
DOWNLOAD_DIR = OUTPUT_DIR / "downloads"

CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CDSE_DOWNLOAD = "https://zipper.dataspace.copernicus.eu/odata/v1"


def get_cdse_token(username: str, password: str) -> str:
    resp = requests.post(
        CDSE_TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def download_product(product_id: str, product_name: str, token: str) -> Path:
    """Download and extract a single product. Returns the .SAFE directory path."""
    safe_dir = DOWNLOAD_DIR / product_name
    if safe_dir.exists():
        log.info(f"  Already extracted: {safe_dir.name}")
        return safe_dir

    zip_path = DOWNLOAD_DIR / f"{product_name}.zip"
    if not zip_path.exists():
        url = f"{CDSE_DOWNLOAD}/Products({product_id})/$value"
        headers = {"Authorization": f"Bearer {token}"}
        log.info(f"  Downloading {product_name} ...")

        with requests.get(url, headers=headers, stream=True, timeout=600) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(
                            f"\r  {pct:5.1f}% ({downloaded / 1e6:.0f}/{total / 1e6:.0f} MB)",
                            end="",
                            flush=True,
                        )
            print()
        log.info(f"  Download complete: {zip_path.name}")
    else:
        log.info(f"  Using cached zip: {zip_path.name}")

    log.info(f"  Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DOWNLOAD_DIR)
    log.info(f"  Extracted: {safe_dir.name}")

    # Remove zip to save disk space
    zip_path.unlink()
    log.info(f"  Removed zip to save space")

    return safe_dir


def main():
    selection_path = OUTPUT_DIR / "iran_selected_products.json"
    if not selection_path.exists():
        log.error(f"Selection file not found: {selection_path}")
        sys.exit(1)

    with open(selection_path) as f:
        selected = json.load(f)

    log.info(f"{'=' * 60}")
    log.info(f"Downloading {len(selected)} Sentinel-1 scenes for Iran")
    log.info(f"{'=' * 60}")

    username = os.environ.get("CDSE_USER")
    password = os.environ.get("CDSE_PASS")
    if not username or not password:
        log.error(
            "CDSE credentials required. Set:\n"
            "  export CDSE_USER='your_email'\n"
            "  export CDSE_PASS='your_password'"
        )
        sys.exit(1)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Authenticating with CDSE ...")
    token = get_cdse_token(username, password)
    log.info("  Authenticated successfully.")

    downloaded = []
    failed = []

    for i, prod in enumerate(selected, 1):
        log.info(f"\n[{i}/{len(selected)}] {prod['name']}")
        log.info(f"  Date: {prod['date']}, {prod['sat']} {prod['direction']}")
        try:
            safe_dir = download_product(prod["id"], prod["name"], token)
            downloaded.append(str(safe_dir))
        except Exception as e:
            log.error(f"  FAILED: {e}")
            failed.append(prod["name"])
            # Re-authenticate in case token expired
            try:
                token = get_cdse_token(username, password)
            except Exception:
                pass

    log.info(f"\n{'=' * 60}")
    log.info(f"Download complete: {len(downloaded)} succeeded, {len(failed)} failed")
    if failed:
        log.warning(f"Failed products: {failed}")
    log.info(f"{'=' * 60}")

    if downloaded:
        log.info("\nNow running RFI detection and map generation...")
        subprocess.run(
            [sys.executable, str(BASE_DIR / "create_map.py")],
            cwd=str(BASE_DIR),
            check=True,
        )

    log.info("\nDone!")


if __name__ == "__main__":
    main()
