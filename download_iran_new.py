#!/usr/bin/env python3
"""
Query CDSE for Iran S1 scenes from Mar 3-8 2026, update the catalog,
and download all missing scenes.
"""
import json, logging, time
from pathlib import Path
from rfi_pipeline import load_credentials, get_cdse_token, download_product
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
CATALOG_PATH = BASE_DIR / "output" / "iran_catalog.json"
DOWNLOAD_DIR = BASE_DIR / "output" / "iran_downloads"

CDSE_CATALOG = "https://catalogue.dataspace.copernicus.eu/odata/v1"

# Iran bounding box (same as used for original catalog)
IRAN_BBOX = "POLYGON((44 25, 63 25, 63 40, 44 40, 44 25))"

def query_cdse_iran(start_date, end_date, skip=0):
    """Query CDSE for S1 GRD scenes over Iran."""
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
    # Load existing catalog
    catalog = json.load(open(CATALOG_PATH))
    existing_names = {p["name"] for p in catalog}
    log.info(f"Existing catalog: {len(catalog)} products")

    # Query for Mar 3–8 (query Mar 2 to catch any late Mar 2 we missed)
    log.info("Querying CDSE for new Iran scenes (Mar 2-8)...")
    all_new = []
    for skip in range(0, 500, 100):
        batch = query_cdse_iran("2026-03-02", "2026-03-09", skip=skip)
        if not batch:
            break
        all_new.extend(batch)
        log.info(f"  Fetched {len(batch)} products (skip={skip})")
        if len(batch) < 100:
            break

    # Filter to truly new
    new_products = [p for p in all_new if p["name"] not in existing_names]
    log.info(f"New products not in catalog: {len(new_products)}")

    if new_products:
        catalog.extend(new_products)
        catalog.sort(key=lambda p: p["start"])
        with open(CATALOG_PATH, "w") as f:
            json.dump(catalog, f, indent=2)
        log.info(f"Updated catalog: {len(catalog)} products total")

    # Now download all missing scenes
    existing_dirs = {d.name.replace(".SAFE", "") for d in DOWNLOAD_DIR.glob("*.SAFE")}
    to_download = [p for p in catalog if p["name"].replace(".SAFE", "") not in existing_dirs
                   and p["start"] >= "2026-03-03"]
    log.info(f"\nScenes to download (Mar 3+): {len(to_download)}")

    if not to_download:
        log.info("Nothing to download!")
        return

    # Show date breakdown
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
    log.info("Authenticated with CDSE")

    downloaded = 0
    failed = 0
    for i, prod in enumerate(to_download, 1):
        name = prod["name"].replace(".SAFE", "")

        # Refresh token every 8 minutes
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
            # Try re-auth on failure
            try:
                token = get_cdse_token(username, password)
                token_time = time.time()
            except Exception:
                pass

        if i % 10 == 0:
            log.info(f"  Progress: {downloaded} downloaded, {failed} failed out of {i}")

    log.info(f"\nDone: {downloaded} downloaded, {failed} failed out of {len(to_download)}")


if __name__ == "__main__":
    main()
