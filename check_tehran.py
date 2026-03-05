#!/usr/bin/env python3
"""
Poll CDSE catalog for new Sentinel-1 products over Tehran.

When new products appear (expected ~Mar 2-3, 2026), downloads them,
runs RFI detection, and regenerates the interactive map.

Usage:
  # One-shot check:
  python check_tehran.py

  # Poll every 2 hours until new data found:
  python check_tehran.py --poll

  # Schedule via cron (every 4 hours):
  # 0 */4 * * * cd ~/sentinel1_rfi_demo && python3 check_tehran.py >> output/check_tehran.log 2>&1
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
DOWNLOAD_DIR = OUTPUT_DIR / "downloads"

CDSE_CATALOG = "https://catalogue.dataspace.copernicus.eu/odata/v1"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CDSE_DOWNLOAD = "https://zipper.dataspace.copernicus.eu/odata/v1"

TEHRAN_CENTER = (51.39, 35.69)

# We already have products up through Feb 19
SEARCH_START = "2026-02-20"
SEARCH_END = "2026-03-10"

# Track what we've already processed
STATE_FILE = OUTPUT_DIR / "tehran_check_state.json"


def load_state() -> set:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return set(json.load(f).get("processed", []))
    return set()


def save_state(processed: set):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({"processed": sorted(processed), "last_check": datetime.utcnow().isoformat()}, f, indent=2)


def search_tehran() -> list:
    wkt_point = f"POINT({TEHRAN_CENTER[0]} {TEHRAN_CENTER[1]})"
    odata_filter = (
        f"Collection/Name eq 'SENTINEL-1' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt_point}') and "
        f"ContentDate/Start gt {SEARCH_START}T00:00:00.000Z and "
        f"ContentDate/Start lt {SEARCH_END}T23:59:59.999Z and "
        f"Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' and "
        f"att/OData.CSC.StringAttribute/Value eq 'GRD')"
    )
    params = {
        "$filter": odata_filter,
        "$orderby": "ContentDate/Start asc",
        "$top": 20,
    }
    log.info(f"Checking CDSE catalog for new Tehran products ({SEARCH_START} to {SEARCH_END})...")
    resp = requests.get(f"{CDSE_CATALOG}/Products", params=params, timeout=30)
    resp.raise_for_status()
    products = resp.json().get("value", [])
    return [{"id": p["Id"], "name": p["Name"], "start": p["ContentDate"]["Start"]} for p in products]


def _load_env():
    """Load credentials from .env file if not already in environment."""
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


def get_token() -> str:
    _load_env()
    username = os.environ.get("CDSE_USER")
    password = os.environ.get("CDSE_PASS")
    if not username or not password:
        log.error("CDSE credentials required: set in .env or export CDSE_USER and CDSE_PASS")
        sys.exit(1)
    resp = requests.post(
        CDSE_TOKEN_URL,
        data={"client_id": "cdse-public", "username": username, "password": password, "grant_type": "password"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def download_and_extract(product_id: str, product_name: str, token: str) -> Path:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_dir = DOWNLOAD_DIR / product_name
    if safe_dir.exists():
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
                        print(f"\r  {downloaded / total * 100:5.1f}%", end="", flush=True)
            print()

    log.info(f"  Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DOWNLOAD_DIR)
    zip_path.unlink()
    return safe_dir


def main():
    parser = argparse.ArgumentParser(description="Check for new Sentinel-1 Tehran data")
    parser.add_argument("--poll", action="store_true", help="Poll every 2 hours until new data found")
    parser.add_argument("--interval", type=int, default=7200, help="Poll interval in seconds (default: 7200 = 2hr)")
    args = parser.parse_args()

    while True:
        products = search_tehran()
        processed = load_state()
        new_products = [p for p in products if p["name"] not in processed]

        if new_products:
            log.info(f"Found {len(new_products)} NEW product(s)!")
            for p in new_products:
                log.info(f"  {p['start'][:16]}  {p['name']}")

            token = get_token()
            for p in new_products:
                download_and_extract(p["id"], p["name"], token)
                processed.add(p["name"])
            save_state(processed)

            log.info("Running RFI detection and map generation...")
            subprocess.run(
                [sys.executable, str(BASE_DIR / "create_map.py")],
                cwd=str(BASE_DIR),
                check=True,
            )
            log.info("Done! New Tehran data processed and map updated.")
            break
        else:
            log.info(f"No new Tehran products yet. (checked {len(products)} existing)")
            save_state(processed)
            if not args.poll:
                log.info("Run with --poll to keep checking automatically.")
                break
            log.info(f"Will check again in {args.interval // 3600}h {(args.interval % 3600) // 60}m ...")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
