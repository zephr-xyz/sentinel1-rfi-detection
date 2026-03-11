#!/usr/bin/env python3
"""
Query CDSE for recent Iran S1 GRD scenes, update catalog, download missing ones.
Used by iran_poller.sh — queries the last 14 days so it catches any new imagery.
Time-limited: downloads for at most MAX_CYCLE_SECONDS so the poller can proceed to
analysis + map + deploy even if there are many products to fetch.
"""
import json, logging, time
from datetime import datetime, timedelta
from pathlib import Path
from rfi_pipeline import load_credentials, get_cdse_token, download_product
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
CATALOG_PATH = BASE_DIR / "output" / "iran_catalog.json"
DOWNLOAD_DIR = BASE_DIR / "output" / "iran_downloads"

CDSE_CATALOG = "https://catalogue.dataspace.copernicus.eu/odata/v1"
IRAN_BBOX = "POLYGON((44 25, 63 25, 63 40, 44 40, 44 25))"

# Only download from the start of the conflict onward
EARLIEST_DATE = "2026-02-28"

# Time limit per cycle so the poller can proceed to analysis/map/deploy
MAX_CYCLE_SECONDS = 30 * 60  # 30 minutes


def query_cdse_iran(start_date, end_date, skip=0):
    odata_filter = (
        f"Collection/Name eq 'SENTINEL-1' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{IRAN_BBOX}') and "
        f"ContentDate/Start gt {start_date}T00:00:00.000Z and "
        f"ContentDate/Start lt {end_date}T23:59:59.999Z and "
        f"Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' and "
        f"att/OData.CSC.StringAttribute/Value eq 'GRD')"
    )
    params = {
        "$filter": odata_filter,
        "$orderby": "ContentDate/Start asc",
        "$top": 100,
        "$skip": skip,
    }
    resp = requests.get(f"{CDSE_CATALOG}/Products", params=params, timeout=60)
    resp.raise_for_status()
    products = resp.json().get("value", [])
    return [{"id": p["Id"], "name": p["Name"], "start": p["ContentDate"]["Start"],
             "end": p["ContentDate"]["End"]} for p in products]


def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Load or create catalog
    if CATALOG_PATH.exists():
        catalog = json.load(open(CATALOG_PATH))
    else:
        catalog = []
    existing_names = {p["name"] for p in catalog}
    log.info(f"Existing catalog: {len(catalog)} products")

    # Query from conflict start to tomorrow
    end_date = datetime.utcnow() + timedelta(days=1)
    start_str = EARLIEST_DATE
    end_str = end_date.strftime("%Y-%m-%d")

    log.info(f"Querying CDSE for Iran scenes ({start_str} to {end_str})...")
    all_results = []
    for skip in range(0, 1000, 100):
        try:
            batch = query_cdse_iran(start_str, end_str, skip=skip)
        except Exception as e:
            log.warning(f"Query failed at skip={skip}: {e}")
            break
        if not batch:
            break
        all_results.extend(batch)
        log.info(f"  Fetched {len(batch)} products (skip={skip})")
        if len(batch) < 100:
            break

    # Add new products to catalog
    new_products = [p for p in all_results if p["name"] not in existing_names]
    log.info(f"New products not in catalog: {len(new_products)}")

    if new_products:
        catalog.extend(new_products)
        catalog.sort(key=lambda p: p["start"])
        with open(CATALOG_PATH, "w") as f:
            json.dump(catalog, f, indent=2)
        log.info(f"Updated catalog: {len(catalog)} products total")

    # Download missing scenes (from conflict start onward, newest first)
    existing_dirs = {d.name.replace(".SAFE", "") for d in DOWNLOAD_DIR.glob("*.SAFE")}
    # Also skip products that have a .zip on disk (failed extract — don't retry endlessly)
    existing_zips = {d.stem for d in DOWNLOAD_DIR.glob("*.zip")}
    existing_all = existing_dirs | existing_zips
    to_download = [p for p in catalog
                   if p["name"].replace(".SAFE", "") not in existing_all
                   and p["start"] >= EARLIEST_DATE]
    to_download.sort(key=lambda p: p["start"], reverse=True)
    log.info(f"Scenes to download: {len(to_download)}")

    if not to_download:
        log.info("Nothing to download!")
        return

    from collections import Counter
    date_counts = Counter(p["start"][:10] for p in to_download)
    for d in sorted(date_counts):
        log.info(f"  {d}: {date_counts[d]} scenes")

    username, password = load_credentials()
    if not username:
        log.error("Missing CDSE credentials in .env")
        return

    token = get_cdse_token(username, password)
    token_time = time.time()
    cycle_start = time.time()
    log.info("Authenticated with CDSE")

    downloaded = 0
    failed = 0
    for i, prod in enumerate(to_download, 1):
        # Time limit — exit gracefully so poller can run analysis + deploy
        elapsed = time.time() - cycle_start
        if elapsed > MAX_CYCLE_SECONDS:
            log.info(f"Time limit reached ({int(elapsed)}s). Stopping downloads, {len(to_download) - i + 1} remaining.")
            break

        name = prod["name"].replace(".SAFE", "")

        if time.time() - token_time > 480:
            try:
                token = get_cdse_token(username, password)
                token_time = time.time()
                log.info("Token refreshed")
            except Exception as e:
                log.warning(f"Token refresh failed: {e}")

        log.info(f"[{i}/{len(to_download)}] {name[:65]} ({prod['start'][:10]})")
        safe_dir = download_product(prod["id"], name, token, DOWNLOAD_DIR)
        if safe_dir:
            downloaded += 1
        else:
            failed += 1
            try:
                token = get_cdse_token(username, password)
                token_time = time.time()
            except Exception:
                pass

        if i % 10 == 0:
            log.info(f"  Progress: {downloaded} downloaded, {failed} failed out of {i}")

    log.info(f"Done: {downloaded} downloaded, {failed} failed out of {i if to_download else 0}")


if __name__ == "__main__":
    main()
