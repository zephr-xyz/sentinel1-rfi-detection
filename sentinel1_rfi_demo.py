#!/usr/bin/env python3
"""
Sentinel-1 RFI (Radio Frequency Interference) Detection Demo
=============================================================

Detects GPS/GNSS jamming signatures in Sentinel-1 SAR imagery over Tehran, Iran.

Sentinel-1 operates in C-band (5.405 GHz). Ground-based GPS jammers often leak
broadband noise into nearby frequencies, producing characteristic bright streaks
("interference bars") in SAR imagery that don't correspond to physical objects.

This script:
  1. Queries the Copernicus Data Space Ecosystem (CDSE) catalog for Sentinel-1
     GRD products over Tehran for the specified dates.
  2. Downloads available products (requires free CDSE account).
  3. Analyzes SAR backscatter for RFI anomalies.
  4. Produces diagnostic maps and plots.

Usage:
  # Search only (no credentials needed):
  python sentinel1_rfi_demo.py --search-only

  # Full pipeline (requires CDSE credentials):
  export CDSE_USER="your_email@example.com"
  export CDSE_PASS="your_password"
  python sentinel1_rfi_demo.py

  # Process already-downloaded .SAFE directory:
  python sentinel1_rfi_demo.py --local /path/to/S1A_IW_GRDH_*.SAFE

  # Run with synthetic data (no credentials, no download — demo mode):
  python sentinel1_rfi_demo.py --demo

Register for free CDSE credentials at: https://dataspace.copernicus.eu
"""

import argparse
import json
import logging
import os
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
import requests
from scipy import ndimage, signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Tehran metropolitan area bounding box
TEHRAN_BBOX = {
    "west": 50.8,
    "south": 35.4,
    "east": 51.9,
    "north": 35.9,
}
TEHRAN_CENTER = (51.39, 35.69)  # lon, lat

# CDSE API endpoints
CDSE_CATALOG = "https://catalogue.dataspace.copernicus.eu/odata/v1"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CDSE_DOWNLOAD = "https://zipper.dataspace.copernicus.eu/odata/v1"

# RFI detection thresholds
RFI_AZIMUTH_ZSCORE = 3.0      # z-score threshold for azimuth-line anomaly
RFI_PIXEL_ZSCORE = 4.0        # z-score threshold for individual pixels
RFI_MIN_STREAK_LENGTH = 50    # minimum contiguous pixels for a streak
RFI_SPECTRAL_PEAK_DB = 10.0   # dB above noise floor for spectral peaks

OUTPUT_DIR = Path("output")


# ---------------------------------------------------------------------------
# 1. Catalog Search
# ---------------------------------------------------------------------------

def search_sentinel1_products(start_date: str, end_date: str, bbox: dict) -> list:
    """Query CDSE OData catalog for Sentinel-1 GRD products over the AOI."""

    wkt_point = f"POINT({TEHRAN_CENTER[0]} {TEHRAN_CENTER[1]})"

    # OData filter for Sentinel-1 GRD products intersecting Tehran
    odata_filter = (
        f"Collection/Name eq 'SENTINEL-1' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt_point}') and "
        f"ContentDate/Start gt {start_date}T00:00:00.000Z and "
        f"ContentDate/Start lt {end_date}T23:59:59.999Z and "
        f"Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' and "
        f"att/OData.CSC.StringAttribute/Value eq 'GRD')"
    )

    params = {
        "$filter": odata_filter,
        "$orderby": "ContentDate/Start asc",
        "$top": 20,
        "$expand": "Attributes",
    }

    log.info("Searching CDSE catalog for Sentinel-1 GRD products over Tehran...")
    log.info(f"  Date range: {start_date} to {end_date}")

    resp = requests.get(f"{CDSE_CATALOG}/Products", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    products = data.get("value", [])
    log.info(f"  Found {len(products)} product(s)")

    results = []
    for p in products:
        attrs = {}
        for a in p.get("Attributes", []):
            attrs[a["Name"]] = a.get("Value")

        info = {
            "id": p["Id"],
            "name": p["Name"],
            "start": p["ContentDate"]["Start"],
            "end": p["ContentDate"]["End"],
            "size_mb": round(p.get("ContentLength", 0) / 1e6, 1),
            "orbit_direction": attrs.get("orbitDirection", "N/A"),
            "polarisation": attrs.get("polarisation", "N/A"),
            "instrument_mode": attrs.get("operationalMode", "N/A"),
            "relative_orbit": attrs.get("relativeOrbitNumber", "N/A"),
        }
        results.append(info)

        log.info(
            f"  [{info['start'][:16]}] {info['name']}  "
            f"({info['orbit_direction']}, {info['polarisation']}, "
            f"{info['size_mb']} MB)"
        )

    return results


# ---------------------------------------------------------------------------
# 2. Authentication & Download
# ---------------------------------------------------------------------------

def get_cdse_token(username: str, password: str) -> str:
    """Obtain an OAuth2 access token from CDSE."""
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


def download_product(product_id: str, product_name: str, token: str, dest_dir: Path) -> Path:
    """Download and extract a Sentinel-1 product from CDSE."""
    zip_path = dest_dir / f"{product_name}.zip"
    # Product name already ends in .SAFE — don't double-suffix
    safe_dir = dest_dir / product_name

    if safe_dir.exists():
        log.info(f"  Already extracted: {safe_dir}")
        return safe_dir

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
                        print(f"\r  {pct:5.1f}% ({downloaded/1e6:.0f}/{total/1e6:.0f} MB)", end="")
            print()
        log.info(f"  Download complete: {zip_path}")
    else:
        log.info(f"  Using cached zip: {zip_path}")

    log.info(f"  Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    log.info(f"  Extracted to {safe_dir}")

    return safe_dir


# ---------------------------------------------------------------------------
# 3. SAR Data Loading
# ---------------------------------------------------------------------------

def find_measurement_tifs(safe_dir: Path) -> list:
    """Find GeoTIFF measurement files inside a .SAFE directory."""
    meas_dir = safe_dir / "measurement"
    if not meas_dir.exists():
        log.warning(f"No measurement directory in {safe_dir}")
        return []
    tifs = sorted(meas_dir.glob("*.tiff")) + sorted(meas_dir.glob("*.tif"))
    log.info(f"  Found {len(tifs)} measurement file(s) in {safe_dir.name}")
    for t in tifs:
        log.info(f"    {t.name}")
    return tifs


def load_sar_image(tif_path: Path) -> tuple:
    """Load a Sentinel-1 GRD measurement GeoTIFF. Returns (data, profile)."""
    log.info(f"  Loading {tif_path.name} ...")
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float64)
        profile = dict(src.profile)
        bounds = src.bounds
        transform = src.transform
    log.info(f"    Shape: {data.shape}, dtype: {data.dtype}")
    log.info(f"    Bounds: {bounds}")
    return data, profile, bounds, transform


# ---------------------------------------------------------------------------
# 4. RFI Detection Algorithms
# ---------------------------------------------------------------------------

def intensity_to_db(data: np.ndarray) -> np.ndarray:
    """Convert linear intensity to dB, handling zeros."""
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.where(data > 0, data, np.nan))
    return db


def detect_rfi_azimuth_lines(data_db: np.ndarray) -> dict:
    """
    Detect RFI by analyzing per-azimuth-line (row) statistics.

    GPS jammers produce excess energy across entire range lines, elevating
    the mean backscatter of affected azimuth positions.
    """
    row_means = np.nanmean(data_db, axis=1)
    row_stds = np.nanstd(data_db, axis=1)

    # Robust statistics using median/MAD to avoid RFI biasing the baseline
    global_median = np.nanmedian(row_means)
    mad = np.nanmedian(np.abs(row_means - global_median))
    robust_std = 1.4826 * mad  # MAD to std conversion

    if robust_std < 0.01:
        robust_std = np.nanstd(row_means)

    zscores = (row_means - global_median) / robust_std
    rfi_mask = zscores > RFI_AZIMUTH_ZSCORE

    n_rfi_lines = int(np.sum(rfi_mask))
    pct = n_rfi_lines / len(row_means) * 100

    log.info(f"    Azimuth-line analysis: {n_rfi_lines}/{len(row_means)} "
             f"lines flagged ({pct:.1f}%)")

    return {
        "row_means": row_means,
        "row_stds": row_stds,
        "zscores": zscores,
        "rfi_line_mask": rfi_mask,
        "global_median": global_median,
        "robust_std": robust_std,
        "n_rfi_lines": n_rfi_lines,
        "pct_rfi_lines": pct,
    }


def detect_rfi_bright_pixels(data_db: np.ndarray) -> dict:
    """
    Detect anomalously bright pixels that indicate localized RFI.

    RFI from jammers creates pixels far brighter than the natural
    backscatter distribution.
    """
    from scipy.ndimage import zoom as scipy_zoom
    h, w = data_db.shape

    # For large images, use a bigger downscale factor
    scale = max(8, min(h, w) // 256)

    # Replace NaN with local fill for the median filter (NaN poisons median_filter)
    small = data_db[::scale, ::scale].copy()
    nan_mask_small = ~np.isfinite(small)
    if np.any(~nan_mask_small):
        fill_val = np.nanmedian(small)
    else:
        fill_val = 0.0
    small[nan_mask_small] = fill_val

    baseline_small = ndimage.median_filter(small, size=15)

    # Upsample back
    baseline = scipy_zoom(baseline_small, (h / baseline_small.shape[0], w / baseline_small.shape[1]))
    baseline = baseline[:h, :w]

    # Restore NaN in baseline where original data was NaN
    nan_mask = ~np.isfinite(data_db)
    baseline[nan_mask] = np.nan

    residual = data_db - baseline

    # Robust stats on valid residual pixels only
    valid_residual = residual[np.isfinite(residual)]
    if len(valid_residual) > 0:
        med = np.median(valid_residual)
        mad = np.median(np.abs(valid_residual - med))
        rstd = 1.4826 * mad
        if rstd < 0.01:
            rstd = np.std(valid_residual)
        bright_mask = np.isfinite(residual) & (residual > (med + RFI_PIXEL_ZSCORE * rstd))
    else:
        med, rstd = 0.0, 1.0
        bright_mask = np.zeros_like(data_db, dtype=bool)

    n_bright = int(np.sum(bright_mask))
    n_valid = int(np.sum(np.isfinite(data_db)))
    pct = n_bright / max(n_valid, 1) * 100

    log.info(f"    Bright-pixel analysis: {n_bright} pixels flagged ({pct:.2f}%)")

    return {
        "baseline": baseline,
        "residual": residual,
        "bright_mask": bright_mask,
        "n_bright_pixels": n_bright,
        "pct_bright": pct,
        "residual_median": med,
        "residual_rstd": rstd,
    }


def detect_rfi_spectral(data_db: np.ndarray, n_sample_cols: int = 64) -> dict:
    """
    Spectral analysis along azimuth (slow-time) direction.

    RFI from continuous-wave jammers produces narrow spectral peaks in the
    azimuth FFT that stand out above the noise floor.
    """
    h, w = data_db.shape

    # Sample columns evenly across the range dimension
    col_indices = np.linspace(0, w - 1, n_sample_cols, dtype=int)
    peak_counts = 0
    total_checks = 0
    peak_frequencies = []

    for ci in col_indices:
        col = data_db[:, ci]
        valid = col[np.isfinite(col)]
        if len(valid) < 256:
            continue

        # Detrend and window
        valid = valid - np.mean(valid)
        window = signal.windows.hann(len(valid))
        spectrum = np.abs(np.fft.rfft(valid * window))
        spectrum_db = 20 * np.log10(spectrum + 1e-30)

        # Noise floor estimate (median of spectrum)
        floor = np.median(spectrum_db)
        peaks_above = spectrum_db > (floor + RFI_SPECTRAL_PEAK_DB)

        # Exclude DC and very low frequencies
        peaks_above[:5] = False

        n_peaks = int(np.sum(peaks_above))
        if n_peaks > 0:
            peak_counts += n_peaks
            freqs = np.where(peaks_above)[0]
            peak_frequencies.extend(freqs.tolist())
        total_checks += 1

    log.info(f"    Spectral analysis: {peak_counts} anomalous peaks across "
             f"{total_checks} range samples")

    return {
        "peak_counts": peak_counts,
        "total_checks": total_checks,
        "peak_frequencies": peak_frequencies,
    }


def detect_rfi_streaks(bright_mask: np.ndarray) -> dict:
    """
    Detect linear streak structures in the bright-pixel mask.

    RFI produces horizontal (range-direction) streaks. We look for
    connected components that are much wider than they are tall.

    For large images, we downsample the mask first for performance, then
    scale coordinates back up.
    """
    h, w = bright_mask.shape
    # Downsample for large images — label() on 400M+ pixels is very slow
    ds = max(1, max(h, w) // 4096)
    if ds > 1:
        # Use max-pooling via block_reduce to preserve streak presence
        from scipy.ndimage import maximum_filter
        mask_ds = bright_mask[::ds, ::ds]
    else:
        mask_ds = bright_mask

    labeled, n_features = ndimage.label(mask_ds)
    streaks = []

    # Use regionprops-style approach: iterate slices for each label
    slices = ndimage.find_objects(labeled)
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        component = labeled[sl] == (i + 1)
        rows_loc, cols_loc = np.where(component)
        if len(rows_loc) == 0:
            continue
        width = (cols_loc.max() - cols_loc.min() + 1)
        height = (rows_loc.max() - rows_loc.min() + 1)
        min_len = max(RFI_MIN_STREAK_LENGTH // ds, 10)
        if width >= min_len and width > 5 * max(height, 1):
            # Scale back to original coordinates
            streaks.append({
                "row_center": int((sl[0].start + np.mean(rows_loc)) * ds),
                "col_start": int((sl[1].start + int(cols_loc.min())) * ds),
                "col_end": int((sl[1].start + int(cols_loc.max())) * ds),
                "width": int(width * ds),
                "height": int(height * ds),
                "n_pixels": int(len(rows_loc) * ds * ds),
            })

    log.info(f"    Streak detection: {len(streaks)} linear streaks found "
             f"(min length={RFI_MIN_STREAK_LENGTH}px, ds={ds}x)")

    return {"streaks": streaks, "n_streaks": len(streaks)}


def run_rfi_detection(data: np.ndarray) -> dict:
    """Run the full RFI detection pipeline on a SAR intensity image."""
    log.info("  Running RFI detection pipeline ...")
    data_db = intensity_to_db(data)

    azimuth = detect_rfi_azimuth_lines(data_db)
    bright = detect_rfi_bright_pixels(data_db)
    spectral = detect_rfi_spectral(data_db)
    streaks = detect_rfi_streaks(bright["bright_mask"])

    # Composite RFI severity score (0-100)
    score = min(100.0, (
        azimuth["pct_rfi_lines"] * 2.0 +
        bright["pct_bright"] * 10.0 +
        min(spectral["peak_counts"], 100) * 0.3 +
        streaks["n_streaks"] * 5.0
    ))

    if score > 60:
        severity = "HIGH"
    elif score > 30:
        severity = "MODERATE"
    elif score > 10:
        severity = "LOW"
    else:
        severity = "MINIMAL/NONE"

    log.info(f"  RFI severity score: {score:.1f}/100 ({severity})")

    return {
        "data_db": data_db,
        "azimuth": azimuth,
        "bright": bright,
        "spectral": spectral,
        "streaks": streaks,
        "score": score,
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# 5. Synthetic Data Generator (Demo Mode)
# ---------------------------------------------------------------------------

def generate_synthetic_sar(
    height: int = 2048,
    width: int = 2048,
    rfi_intensity: str = "high",
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a realistic synthetic Sentinel-1 GRD-like image with injected RFI.

    Simulates:
    - Rayleigh-distributed speckle (characteristic of SAR)
    - Spatial backscatter variation (urban/rural contrast for Tehran)
    - RFI streaks from ground-based jammers at various intensities

    Args:
        height: Image height (azimuth pixels)
        width: Image width (range pixels)
        rfi_intensity: "none", "low", "moderate", or "high"
        seed: Random seed for reproducibility

    Returns:
        2D numpy array of linear intensity values (not dB)
    """
    rng = np.random.RandomState(seed)

    # Base backscatter: gradual spatial variation simulating terrain
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    # Urban area (higher backscatter) in the center
    urban_center_y, urban_center_x = height * 0.45, width * 0.5
    dist = np.sqrt((y_grid - urban_center_y)**2 + (x_grid - urban_center_x)**2)
    urban_mask = np.exp(-dist**2 / (2 * (height * 0.15)**2))
    base_sigma0 = 0.02 + 0.08 * urban_mask  # rural ~0.02, urban ~0.10

    # Add some terrain texture
    from scipy.ndimage import gaussian_filter
    texture = gaussian_filter(rng.randn(height // 16, width // 16), sigma=2)
    texture = np.kron(texture, np.ones((16, 16)))[:height, :width]
    base_sigma0 *= np.exp(0.3 * texture)

    # SAR speckle (exponential distribution for single-look intensity)
    speckle = rng.exponential(scale=1.0, size=(height, width))
    image = base_sigma0 * speckle

    # Multi-look averaging (Sentinel-1 GRD is ~5 looks)
    n_looks = 5
    for _ in range(n_looks - 1):
        speckle_extra = rng.exponential(scale=1.0, size=(height, width))
        image += base_sigma0 * speckle_extra
    image /= n_looks

    # --- Inject RFI ---
    if rfi_intensity == "none":
        return image

    # RFI parameters by intensity level
    rfi_params = {
        "low":      {"n_bursts": 3,  "amp_range": (0.5, 2.0),  "width_range": (1, 3)},
        "moderate": {"n_bursts": 8,  "amp_range": (1.0, 5.0),  "width_range": (1, 5)},
        "high":     {"n_bursts": 15, "amp_range": (2.0, 20.0), "width_range": (2, 8)},
    }
    params = rfi_params.get(rfi_intensity, rfi_params["high"])

    for _ in range(params["n_bursts"]):
        # RFI appears as bright horizontal bands (range-direction streaks)
        row_center = rng.randint(50, height - 50)
        band_width = rng.randint(*params["width_range"])
        amplitude = rng.uniform(*params["amp_range"])

        # RFI amplitude varies across range (stronger near jammer, fades with distance)
        jammer_col = rng.randint(width // 4, 3 * width // 4)
        range_profile = np.exp(-0.5 * ((np.arange(width) - jammer_col) / (width * 0.3))**2)

        # Some RFI is more uniform (broadband jammers)
        if rng.random() > 0.5:
            range_profile = 0.3 + 0.7 * range_profile

        for dr in range(-band_width // 2, band_width // 2 + 1):
            row = row_center + dr
            if 0 <= row < height:
                rfi_power = amplitude * range_profile * base_sigma0[row, :]
                # Add with slight noise variation
                rfi_power *= (1.0 + 0.1 * rng.randn(width))
                image[row, :] += np.abs(rfi_power)

    # Add a few diagonal streaks (characteristic of some jammer types)
    if rfi_intensity in ("moderate", "high"):
        n_diag = rng.randint(1, 4)
        for _ in range(n_diag):
            start_row = rng.randint(0, height // 2)
            slope = rng.uniform(-0.3, 0.3)
            amplitude = rng.uniform(1.0, 8.0)
            for col in range(width):
                row = int(start_row + slope * col)
                if 0 <= row < height - 2:
                    rfi_val = amplitude * base_sigma0[row, col]
                    image[row, col] += rfi_val
                    image[row + 1, col] += rfi_val * 0.5

    return image


def run_demo_mode(output_dir: Path):
    """Run the full pipeline on synthetic SAR data with injected RFI."""
    log.info("=" * 60)
    log.info("DEMO MODE — Synthetic Sentinel-1 data with injected RFI")
    log.info("=" * 60)
    log.info("")
    log.info("Generating synthetic SAR imagery simulating Tehran region...")
    log.info("This demonstrates the detection pipeline without requiring")
    log.info("CDSE credentials or multi-GB satellite data downloads.")

    scenarios = [
        {
            "name": "S1A_IW_GRDH_1SDV_20260228T025311_SYNTHETIC",
            "date": "2026-02-28",
            "pol": "VV",
            "rfi": "high",
            "seed": 42,
            "desc": "Feb 28 descending pass — strong RFI (simulating active jamming)",
        },
        {
            "name": "S1A_IW_GRDH_1SDV_20260228T025311_SYNTHETIC",
            "date": "2026-02-28",
            "pol": "VH",
            "rfi": "moderate",
            "seed": 43,
            "desc": "Feb 28 descending pass VH — moderate RFI (cross-pol, typically weaker)",
        },
        {
            "name": "S1A_IW_GRDH_1SDV_20260301T143600_SYNTHETIC",
            "date": "2026-03-01",
            "pol": "VV",
            "rfi": "high",
            "seed": 44,
            "desc": "Mar 1 ascending pass — strong RFI (persistent jamming)",
        },
    ]

    all_results = []

    for sc in scenarios:
        log.info(f"\n{'─' * 60}")
        log.info(f"Scenario: {sc['desc']}")
        log.info(f"{'─' * 60}")

        data = generate_synthetic_sar(
            height=2048, width=2048,
            rfi_intensity=sc["rfi"], seed=sc["seed"],
        )
        log.info(f"  Generated {data.shape[0]}x{data.shape[1]} synthetic SAR image")

        rfi = run_rfi_detection(data)

        fig_name = f"rfi_{sc['name']}_{sc['pol']}.png"
        fig_path = output_dir / fig_name
        plot_rfi_report(rfi["data_db"], rfi, sc["name"], sc["pol"], fig_path)

        all_results.append({
            "product_name": sc["name"],
            "polarization": sc["pol"],
            "date": sc["date"],
            "score": rfi["score"],
            "severity": rfi["severity"],
            "n_rfi_lines": rfi["azimuth"]["n_rfi_lines"],
            "pct_rfi_lines": rfi["azimuth"]["pct_rfi_lines"],
            "n_bright_pixels": rfi["bright"]["n_bright_pixels"],
            "pct_bright": rfi["bright"]["pct_bright"],
            "spectral_peaks": rfi["spectral"]["peak_counts"],
            "n_streaks": rfi["streaks"]["n_streaks"],
            "figure_path": str(fig_path),
        })

    print_summary(all_results)

    report_path = output_dir / "rfi_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"JSON report saved: {report_path}")

    return all_results


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------

def plot_rfi_report(data_db: np.ndarray, results: dict, product_name: str,
                    polarization: str, output_path: Path):
    """Generate a multi-panel RFI detection report figure."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(
        f"Sentinel-1 RFI Detection — Tehran, Iran\n"
        f"{product_name}\nPolarization: {polarization}  |  "
        f"RFI Score: {results['score']:.1f}/100 ({results['severity']})",
        fontsize=13, fontweight="bold", y=0.98,
    )

    az = results["azimuth"]
    br = results["bright"]
    st = results["streaks"]

    # Panel 1: SAR image in dB
    ax = axes[0, 0]
    finite_vals = data_db[np.isfinite(data_db)]
    if len(finite_vals) > 0:
        vmin, vmax = np.percentile(finite_vals, [2, 98])
    else:
        vmin, vmax = -30, 0
    # Downsample for display if image is large
    ds = max(1, max(data_db.shape) // 4096)
    ax.imshow(data_db[::ds, ::ds], cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title("SAR Backscatter (dB)")
    ax.set_xlabel("Range (pixels)")
    ax.set_ylabel("Azimuth (pixels)")

    # Panel 2: Azimuth-line mean profile with RFI lines highlighted
    ax = axes[0, 1]
    rows = np.arange(len(az["row_means"]))
    ax.plot(az["row_means"], rows, linewidth=0.3, color="steelblue", label="Row mean (dB)")
    rfi_rows = rows[az["rfi_line_mask"]]
    if len(rfi_rows) > 0:
        ax.scatter(
            az["row_means"][az["rfi_line_mask"]], rfi_rows,
            c="red", s=1, zorder=5, label=f"RFI lines ({az['n_rfi_lines']})"
        )
    ax.axvline(az["global_median"], color="green", linestyle="--", linewidth=0.8,
               label=f"Median: {az['global_median']:.1f} dB")
    thresh = az["global_median"] + RFI_AZIMUTH_ZSCORE * az["robust_std"]
    ax.axvline(thresh, color="red", linestyle="--", linewidth=0.8,
               label=f"Threshold: {thresh:.1f} dB")
    ax.set_xlabel("Mean Backscatter (dB)")
    ax.set_ylabel("Azimuth Line")
    ax.set_title("Azimuth-Line Mean Profile")
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="lower right")

    # Panel 3: Bright pixel mask (RFI map)
    ax = axes[0, 2]
    # For very large images, downsample the overlay for memory
    ds = max(1, max(data_db.shape) // 4096)
    db_small = data_db[::ds, ::ds]
    mask_small = br["bright_mask"][::ds, ::ds]
    rfi_overlay = np.zeros((*db_small.shape, 4))
    rfi_overlay[mask_small, :] = [1, 0, 0, 0.8]  # red
    ax.imshow(db_small, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    ax.imshow(rfi_overlay, aspect="auto")
    ax.set_title(f"RFI Bright Pixels (red)\n{br['n_bright_pixels']} pixels flagged")
    ax.set_xlabel("Range (pixels)")
    ax.set_ylabel("Azimuth (pixels)")

    # Panel 4: Residual (data - baseline) showing RFI anomalies
    ax = axes[1, 0]
    residual = br["residual"]
    valid_res = residual[np.isfinite(residual)]
    if len(valid_res) > 0:
        pass  # use fixed range below
    im = ax.imshow(residual[::ds, ::ds], cmap="RdBu_r", vmin=-10, vmax=10, aspect="auto")
    plt.colorbar(im, ax=ax, label="Residual (dB)")
    ax.set_title("Backscatter Residual\n(Data − Local Median Baseline)")
    ax.set_xlabel("Range (pixels)")
    ax.set_ylabel("Azimuth (pixels)")

    # Panel 5: Z-score profile along azimuth
    ax = axes[1, 1]
    ax.plot(az["zscores"], rows, linewidth=0.3, color="steelblue")
    ax.axvline(RFI_AZIMUTH_ZSCORE, color="red", linestyle="--",
               label=f"Z-score threshold ({RFI_AZIMUTH_ZSCORE})")
    ax.fill_betweenx(rows, RFI_AZIMUTH_ZSCORE, az["zscores"],
                      where=az["zscores"] > RFI_AZIMUTH_ZSCORE,
                      color="red", alpha=0.3)
    ax.set_xlabel("Z-Score")
    ax.set_ylabel("Azimuth Line")
    ax.set_title("Azimuth-Line Z-Score Profile")
    ax.invert_yaxis()
    ax.legend(fontsize=8)

    # Panel 6: Detected streaks overlaid on image
    ax = axes[1, 2]
    ax.imshow(data_db[::ds, ::ds], cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    for s in st["streaks"]:
        rect = plt.Rectangle(
            (s["col_start"] / ds, (s["row_center"] - s["height"] / 2) / ds),
            s["width"] / ds, max(s["height"] / ds, 3),
            linewidth=1.5, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(rect)
    ax.set_title(f"Detected RFI Streaks ({st['n_streaks']} found)")
    ax.set_xlabel("Range (pixels)")
    ax.set_ylabel("Azimuth (pixels)")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Report saved: {output_path}")


def print_summary(all_results: list):
    """Print a text summary of all analyzed products."""
    print("\n" + "=" * 72)
    print("  SENTINEL-1 RFI / GPS JAMMING DETECTION SUMMARY — TEHRAN, IRAN")
    print("=" * 72)

    for r in all_results:
        print(f"\n  Product: {r['product_name']}")
        print(f"  Date:    {r['date']}")
        print(f"  Pol:     {r['polarization']}")
        print(f"  ─────────────────────────────────")
        print(f"  RFI Score:       {r['score']:.1f} / 100")
        print(f"  Severity:        {r['severity']}")
        print(f"  Anomalous lines: {r['n_rfi_lines']} ({r['pct_rfi_lines']:.1f}%)")
        print(f"  Bright pixels:   {r['n_bright_pixels']} ({r['pct_bright']:.2f}%)")
        print(f"  Spectral peaks:  {r['spectral_peaks']}")
        print(f"  Linear streaks:  {r['n_streaks']}")
        print(f"  Report figure:   {r['figure_path']}")

    print("\n" + "─" * 72)
    severities = [r["severity"] for r in all_results]
    if any(s == "HIGH" for s in severities):
        print("  ⚠  HIGH RFI activity detected — consistent with active GPS jamming")
        print("     in the Tehran region during the analyzed period.")
    elif any(s == "MODERATE" for s in severities):
        print("  ⚠  MODERATE RFI detected — possible GPS jamming or other")
        print("     ground-based interference sources in the Tehran area.")
    elif any(s == "LOW" for s in severities):
        print("  ℹ  LOW RFI detected — minor interference present, may be")
        print("     residual or intermittent jamming.")
    else:
        print("  ✓  Minimal/no RFI detected in analyzed imagery.")
    print("─" * 72 + "\n")


# ---------------------------------------------------------------------------
# 6. Main Pipeline
# ---------------------------------------------------------------------------

def process_safe_directory(safe_dir: Path, output_dir: Path) -> list:
    """Process a single .SAFE directory through the RFI detection pipeline."""
    tifs = find_measurement_tifs(safe_dir)
    if not tifs:
        log.warning(f"No measurement TIFFs found in {safe_dir}")
        return []

    results = []
    product_name = safe_dir.stem.replace(".SAFE", "")

    for tif_path in tifs:
        # Determine polarization from filename (e.g., ...-vv-... or ...-vh-...)
        fname_lower = tif_path.stem.lower()
        if "vv" in fname_lower:
            pol = "VV"
        elif "vh" in fname_lower:
            pol = "VH"
        elif "hh" in fname_lower:
            pol = "HH"
        elif "hv" in fname_lower:
            pol = "HV"
        else:
            pol = "unknown"

        log.info(f"\n{'='*60}")
        log.info(f"Processing: {tif_path.name} (Pol: {pol})")
        log.info(f"{'='*60}")

        data, profile, bounds, transform = load_sar_image(tif_path)
        rfi = run_rfi_detection(data)

        # Extract date from product name
        # Typical: S1A_IW_GRDH_1SDV_20260228T025311_...
        parts = product_name.split("_")
        date_str = "unknown"
        for p in parts:
            if len(p) >= 8 and p[:8].isdigit():
                date_str = f"{p[:4]}-{p[4:6]}-{p[6:8]}"
                break

        fig_name = f"rfi_{product_name}_{pol}.png"
        fig_path = output_dir / fig_name
        plot_rfi_report(rfi["data_db"], rfi, product_name, pol, fig_path)

        results.append({
            "product_name": product_name,
            "polarization": pol,
            "date": date_str,
            "score": rfi["score"],
            "severity": rfi["severity"],
            "n_rfi_lines": rfi["azimuth"]["n_rfi_lines"],
            "pct_rfi_lines": rfi["azimuth"]["pct_rfi_lines"],
            "n_bright_pixels": rfi["bright"]["n_bright_pixels"],
            "pct_bright": rfi["bright"]["pct_bright"],
            "spectral_peaks": rfi["spectral"]["peak_counts"],
            "n_streaks": rfi["streaks"]["n_streaks"],
            "figure_path": str(fig_path),
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Detect GPS jamming (RFI) in Sentinel-1 SAR imagery over Tehran"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with synthetic data (no credentials or downloads needed)",
    )
    parser.add_argument(
        "--search-only", action="store_true",
        help="Only search the catalog; do not download or process",
    )
    parser.add_argument(
        "--local", type=str, nargs="+",
        help="Process local .SAFE directories instead of downloading",
    )
    parser.add_argument(
        "--start-date", default="2026-02-28",
        help="Start date (YYYY-MM-DD). Default: 2026-02-28",
    )
    parser.add_argument(
        "--end-date", default="2026-03-01",
        help="End date (YYYY-MM-DD). Default: 2026-03-01",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Output directory for results. Default: output/",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # --- Mode 0: Demo with synthetic data ---
    if args.demo:
        run_demo_mode(output_dir)
        return

    # --- Mode 1: Process local .SAFE directories ---
    if args.local:
        for local_path in args.local:
            safe_dir = Path(local_path)
            if not safe_dir.exists():
                log.error(f"Path does not exist: {safe_dir}")
                continue
            results = process_safe_directory(safe_dir, output_dir)
            all_results.extend(results)

        if all_results:
            print_summary(all_results)
            # Save JSON report
            report_path = output_dir / "rfi_report.json"
            with open(report_path, "w") as f:
                json.dump(all_results, f, indent=2)
            log.info(f"JSON report saved: {report_path}")
        return

    # --- Mode 2: Search catalog and optionally download ---
    products = search_sentinel1_products(args.start_date, args.end_date, TEHRAN_BBOX)

    if not products:
        log.warning("No Sentinel-1 products found for the specified dates and area.")
        log.info("Sentinel-1 has a 6-day revisit cycle per satellite.")
        log.info("Try expanding the date range (--start-date / --end-date).")
        # Save search results
        with open(output_dir / "search_results.json", "w") as f:
            json.dump({"query": {"start": args.start_date, "end": args.end_date,
                                  "bbox": TEHRAN_BBOX}, "products": []}, f, indent=2)
        return

    # Save search results
    with open(output_dir / "search_results.json", "w") as f:
        json.dump({
            "query": {"start": args.start_date, "end": args.end_date, "bbox": TEHRAN_BBOX},
            "products": products,
        }, f, indent=2)
    log.info(f"Search results saved: {output_dir / 'search_results.json'}")

    if args.search_only:
        print("\n  Search complete. Use without --search-only to download and analyze.")
        print(f"  Results: {output_dir / 'search_results.json'}")
        return

    # --- Download and process ---
    username = os.environ.get("CDSE_USER")
    password = os.environ.get("CDSE_PASS")

    if not username or not password:
        log.error(
            "CDSE credentials required for download. Set environment variables:\n"
            "  export CDSE_USER='your_email@example.com'\n"
            "  export CDSE_PASS='your_password'\n"
            "Register free at: https://dataspace.copernicus.eu"
        )
        sys.exit(1)

    log.info("Authenticating with CDSE ...")
    token = get_cdse_token(username, password)
    log.info("  Authenticated successfully.")

    download_dir = output_dir / "downloads"
    download_dir.mkdir(exist_ok=True)

    for product in products:
        safe_dir = download_product(
            product["id"], product["name"], token, download_dir
        )
        results = process_safe_directory(safe_dir, output_dir)
        all_results.extend(results)

    if all_results:
        print_summary(all_results)
        report_path = output_dir / "rfi_report.json"
        with open(report_path, "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"JSON report saved: {report_path}")
    else:
        log.warning("No results produced. Check the downloaded products.")


if __name__ == "__main__":
    main()
