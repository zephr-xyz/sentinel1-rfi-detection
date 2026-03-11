#!/usr/bin/env python3
"""
NISAR L-band RFI (Radio Frequency Interference) Detection Demo
===============================================================

Detects GPS/GNSS jamming signatures in NISAR L-band SAR imagery.

NISAR operates at L-band (1.257 GHz / 24 cm wavelength), which is only ~30 MHz
from the GPS L2 frequency (1.2276 GHz). This makes NISAR dramatically more
sensitive to GPS jamming interference than Sentinel-1's C-band (5.405 GHz).

This script:
  1. Searches NASA CMR for NISAR RSLC products over a region of interest.
  2. Downloads products via earthaccess (requires free NASA Earthdata account).
  3. Reads complex SLC data from HDF5, converts to intensity.
  4. Runs the same RFI detection algorithms used for Sentinel-1.
  5. Produces diagnostic maps and plots.

Usage:
  # Search only (requires Earthdata login):
  python nisar_rfi_demo.py --search-only

  # Full pipeline:
  python nisar_rfi_demo.py

  # Process already-downloaded HDF5 file:
  python nisar_rfi_demo.py --local /path/to/NISAR_L1_PR_RSLC_*.h5

  # Specify date range:
  python nisar_rfi_demo.py --start-date 2026-01-01 --end-date 2026-01-31

Prerequisites:
  pip install h5py earthaccess numpy scipy matplotlib rasterio
  # Then run: earthaccess login  (or set EARTHDATA_USERNAME / EARTHDATA_PASSWORD)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
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

TEHRAN_BBOX = {
    "west": 50.8,
    "south": 35.4,
    "east": 51.9,
    "north": 35.9,
}

# NASA CMR collection concept ID for NISAR RSLC Beta V1
NISAR_RSLC_COLLECTION = "NISAR_L1_RSLC_BETA_V1"
NISAR_RSLC_CONCEPT_ID = "C2850225585-ASF"

# RFI detection thresholds (same as Sentinel-1 pipeline)
RFI_AZIMUTH_ZSCORE = 3.0
RFI_PIXEL_ZSCORE = 4.0
RFI_MIN_STREAK_LENGTH = 50
RFI_SPECTRAL_PEAK_DB = 10.0

OUTPUT_DIR = Path("output")

# NISAR HDF5 paths
NISAR_SLC_GROUP = "/science/LSAR/RSLC/swaths"
NISAR_FREQ_A = "frequencyA"
NISAR_ID_GROUP = "/science/LSAR/identification"


# ---------------------------------------------------------------------------
# 1. Catalog Search via earthaccess / CMR
# ---------------------------------------------------------------------------

def search_nisar_products(start_date: str, end_date: str, bbox: dict) -> list:
    """Search NASA CMR for NISAR RSLC products over the AOI."""
    import earthaccess

    log.info("Searching NASA CMR for NISAR RSLC products...")
    log.info(f"  Date range: {start_date} to {end_date}")
    log.info(f"  Bounding box: {bbox}")

    try:
        earthaccess.login(strategy="environment")
    except Exception:
        try:
            earthaccess.login(strategy="netrc")
        except Exception:
            log.warning("Not logged in to Earthdata. Search still works but download won't.")

    results = earthaccess.search_data(
        concept_id=NISAR_RSLC_CONCEPT_ID,
        temporal=(start_date, end_date),
        bounding_box=(bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
    )

    log.info(f"  Found {len(results)} granule(s)")

    products = []
    for g in results:
        umm = g.get("umm", g) if isinstance(g, dict) else {}
        native_id = g.get("meta", {}).get("native-id", "") if isinstance(g, dict) else ""

        # earthaccess granule objects have helper methods
        try:
            name = g["meta"]["native-id"]
        except (KeyError, TypeError):
            name = str(g)

        try:
            size_mb = g.size()
        except Exception:
            size_mb = 0

        try:
            links = g.data_links()
        except Exception:
            links = []

        info = {
            "name": name,
            "size_mb": round(size_mb, 1),
            "data_links": links,
            "granule": g,
        }
        products.append(info)
        log.info(f"  {name} ({size_mb:.0f} MB)")

    return products


# ---------------------------------------------------------------------------
# 2. Download
# ---------------------------------------------------------------------------

def download_nisar_products(products: list, dest_dir: Path) -> list:
    """Download NISAR products using earthaccess."""
    import earthaccess

    dest_dir.mkdir(parents=True, exist_ok=True)

    granules = [p["granule"] for p in products]

    # Check for already-downloaded files
    to_download = []
    existing_paths = []
    for p in products:
        name = p["name"]
        # Look for existing .h5 files matching this granule
        matches = list(dest_dir.glob(f"{name}*"))
        if matches:
            log.info(f"  Already downloaded: {matches[0].name}")
            existing_paths.append(matches[0])
        else:
            to_download.append(p["granule"])

    if to_download:
        log.info(f"  Downloading {len(to_download)} product(s) to {dest_dir}...")
        earthaccess.login(strategy="environment")
        downloaded = earthaccess.download(to_download, str(dest_dir))
        all_paths = existing_paths + [Path(p) for p in downloaded]
    else:
        all_paths = existing_paths

    return all_paths


# ---------------------------------------------------------------------------
# 3. HDF5 Data Loading
# ---------------------------------------------------------------------------

def explore_h5_structure(h5_path: Path):
    """Print the HDF5 group/dataset structure for debugging."""
    import h5py

    def _print_tree(name, obj):
        if isinstance(obj, h5py.Dataset):
            log.info(f"  DS  {name}  shape={obj.shape}  dtype={obj.dtype}")
        else:
            log.info(f"  GRP {name}/")

    log.info(f"HDF5 structure of {h5_path.name}:")
    with h5py.File(h5_path, "r") as f:
        f.visititems(_print_tree)


def find_slc_datasets(h5_path: Path) -> list:
    """Find all SLC polarization datasets in a NISAR RSLC HDF5 file."""
    import h5py

    datasets = []
    seen_paths = set()
    with h5py.File(h5_path, "r") as f:
        # Try standard NISAR RSLC path structure
        for base in [NISAR_SLC_GROUP, "/science/LSAR/RSLC/swaths",
                      "/science/LSAR/SLC/swaths"]:
            if base not in f:
                continue
            grp = f[base]
            for freq_name in ["frequencyA", "frequencyB"]:
                if freq_name not in grp:
                    continue
                freq_grp = grp[freq_name]
                for pol in ["HH", "HV", "VV", "VH"]:
                    if pol in freq_grp and isinstance(freq_grp[pol], h5py.Dataset):
                        ds_path = f"{base}/{freq_name}/{pol}"
                        if ds_path in seen_paths:
                            continue
                        seen_paths.add(ds_path)
                        datasets.append({
                            "path": ds_path,
                            "frequency": freq_name,
                            "polarization": pol,
                            "shape": freq_grp[pol].shape,
                            "dtype": str(freq_grp[pol].dtype),
                        })

        # If nothing found, search more broadly
        if not datasets:
            log.warning("Standard RSLC paths not found, searching HDF5 tree...")

            def _find_complex(name, obj):
                if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.complexfloating):
                    datasets.append({
                        "path": name,
                        "frequency": "unknown",
                        "polarization": name.split("/")[-1],
                        "shape": obj.shape,
                        "dtype": str(obj.dtype),
                    })
            f.visititems(_find_complex)

    for ds in datasets:
        log.info(f"  Found SLC: {ds['path']}  shape={ds['shape']}  dtype={ds['dtype']}")

    return datasets


def load_nisar_slc(h5_path: Path, dataset_path: str, subsample: int = 1) -> tuple:
    """
    Load NISAR RSLC complex data and convert to intensity.

    Returns (intensity, metadata_dict).
    """
    import h5py

    log.info(f"  Loading {dataset_path} from {h5_path.name}...")

    with h5py.File(h5_path, "r") as f:
        ds = f[dataset_path]
        full_shape = ds.shape
        log.info(f"    Full shape: {full_shape}, dtype: {ds.dtype}")

        if subsample > 1:
            # Read every Nth sample in both dimensions
            slc = ds[::subsample, ::subsample]
        else:
            slc = ds[()]

        # Extract geolocation if available
        meta = {}
        for id_path in [NISAR_ID_GROUP, "/science/LSAR/identification"]:
            if id_path in f:
                id_grp = f[id_path]
                for key in ["absoluteOrbitNumber", "trackNumber", "frameNumber",
                             "lookDirection", "orbitPassDirection",
                             "zeroDopplerStartTime", "zeroDopplerEndTime",
                             "boundingPolygon"]:
                    if key in id_grp:
                        val = id_grp[key][()]
                        if isinstance(val, bytes):
                            val = val.decode("utf-8")
                        elif isinstance(val, np.ndarray) and val.size == 1:
                            val = val.item()
                        meta[key] = val
                break

        # Try to get geographic coordinates
        for coord_base in ["/science/LSAR/RSLC/metadata/geolocationGrid",
                            "/science/LSAR/RSLC/swaths/frequencyA"]:
            if coord_base in f:
                cgrp = f[coord_base]
                for lat_key in ["latitude", "coordinateY", "yCoordinates"]:
                    if lat_key in cgrp:
                        lat_data = cgrp[lat_key][()]
                        meta["lat_grid"] = lat_data
                        break
                for lon_key in ["longitude", "coordinateX", "xCoordinates"]:
                    if lon_key in cgrp:
                        lon_data = cgrp[lon_key][()]
                        meta["lon_grid"] = lon_data
                        break
                if "lat_grid" in meta:
                    break

    log.info(f"    Loaded SLC shape: {slc.shape}")

    # Convert complex SLC to intensity: |I + jQ|^2
    intensity = np.abs(slc).astype(np.float64) ** 2
    log.info(f"    Intensity range: {np.nanmin(intensity):.4e} - {np.nanmax(intensity):.4e}")

    return intensity, meta, full_shape


# ---------------------------------------------------------------------------
# 4. RFI Detection (reused from sentinel1_rfi_demo.py)
# ---------------------------------------------------------------------------

def intensity_to_db(data: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.where(data > 0, data, np.nan))
    return db


def detect_rfi_azimuth_lines(data_db: np.ndarray) -> dict:
    row_means = np.nanmean(data_db, axis=1)
    row_stds = np.nanstd(data_db, axis=1)
    global_median = np.nanmedian(row_means)
    mad = np.nanmedian(np.abs(row_means - global_median))
    robust_std = 1.4826 * mad
    if robust_std < 0.01:
        robust_std = np.nanstd(row_means)
    zscores = (row_means - global_median) / robust_std
    rfi_mask = zscores > RFI_AZIMUTH_ZSCORE
    n_rfi_lines = int(np.sum(rfi_mask))
    pct = n_rfi_lines / len(row_means) * 100
    log.info(f"    Azimuth-line analysis: {n_rfi_lines}/{len(row_means)} lines flagged ({pct:.1f}%)")
    return {
        "row_means": row_means, "row_stds": row_stds, "zscores": zscores,
        "rfi_line_mask": rfi_mask, "global_median": global_median,
        "robust_std": robust_std, "n_rfi_lines": n_rfi_lines, "pct_rfi_lines": pct,
    }


def detect_rfi_bright_pixels(data_db: np.ndarray) -> dict:
    from scipy.ndimage import zoom as scipy_zoom
    h, w = data_db.shape
    scale = max(8, min(h, w) // 256)
    small = data_db[::scale, ::scale].copy()
    nan_mask_small = ~np.isfinite(small)
    fill_val = np.nanmedian(small) if np.any(~nan_mask_small) else 0.0
    small[nan_mask_small] = fill_val
    baseline_small = ndimage.median_filter(small, size=15)
    baseline = scipy_zoom(baseline_small, (h / baseline_small.shape[0], w / baseline_small.shape[1]))
    baseline = baseline[:h, :w]
    nan_mask = ~np.isfinite(data_db)
    baseline[nan_mask] = np.nan
    residual = data_db - baseline
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
        "baseline": baseline, "residual": residual, "bright_mask": bright_mask,
        "n_bright_pixels": n_bright, "pct_bright": pct,
        "residual_median": med, "residual_rstd": rstd,
    }


def detect_rfi_spectral(data_db: np.ndarray, n_sample_cols: int = 64) -> dict:
    h, w = data_db.shape
    col_indices = np.linspace(0, w - 1, n_sample_cols, dtype=int)
    peak_counts = 0
    total_checks = 0
    peak_frequencies = []
    for ci in col_indices:
        col = data_db[:, ci]
        valid = col[np.isfinite(col)]
        if len(valid) < 256:
            continue
        valid = valid - np.mean(valid)
        window = signal.windows.hann(len(valid))
        spectrum = np.abs(np.fft.rfft(valid * window))
        spectrum_db = 20 * np.log10(spectrum + 1e-30)
        floor = np.median(spectrum_db)
        peaks_above = spectrum_db > (floor + RFI_SPECTRAL_PEAK_DB)
        peaks_above[:5] = False
        n_peaks = int(np.sum(peaks_above))
        if n_peaks > 0:
            peak_counts += n_peaks
            peak_frequencies.extend(np.where(peaks_above)[0].tolist())
        total_checks += 1
    log.info(f"    Spectral analysis: {peak_counts} anomalous peaks across {total_checks} range samples")
    return {"peak_counts": peak_counts, "total_checks": total_checks, "peak_frequencies": peak_frequencies}


def detect_rfi_streaks(bright_mask: np.ndarray) -> dict:
    h, w = bright_mask.shape
    ds = max(1, max(h, w) // 4096)
    mask_ds = bright_mask[::ds, ::ds] if ds > 1 else bright_mask
    labeled, n_features = ndimage.label(mask_ds)
    streaks = []
    slices = ndimage.find_objects(labeled)
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        component = labeled[sl] == (i + 1)
        rows_loc, cols_loc = np.where(component)
        if len(rows_loc) == 0:
            continue
        width = cols_loc.max() - cols_loc.min() + 1
        height = rows_loc.max() - rows_loc.min() + 1
        min_len = max(RFI_MIN_STREAK_LENGTH // ds, 10)
        if width >= min_len and width > 5 * max(height, 1):
            streaks.append({
                "row_center": int((sl[0].start + np.mean(rows_loc)) * ds),
                "col_start": int((sl[1].start + int(cols_loc.min())) * ds),
                "col_end": int((sl[1].start + int(cols_loc.max())) * ds),
                "width": int(width * ds), "height": int(height * ds),
                "n_pixels": int(len(rows_loc) * ds * ds),
            })
    log.info(f"    Streak detection: {len(streaks)} linear streaks found")
    return {"streaks": streaks, "n_streaks": len(streaks)}


def run_rfi_detection(data: np.ndarray) -> dict:
    log.info("  Running RFI detection pipeline...")
    data_db = intensity_to_db(data)
    azimuth = detect_rfi_azimuth_lines(data_db)
    bright = detect_rfi_bright_pixels(data_db)
    spectral = detect_rfi_spectral(data_db)
    streaks = detect_rfi_streaks(bright["bright_mask"])
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
        "data_db": data_db, "azimuth": azimuth, "bright": bright,
        "spectral": spectral, "streaks": streaks, "score": score, "severity": severity,
    }


# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------

def plot_rfi_report(data_db, results, product_name, polarization, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(
        f"NISAR L-band RFI Detection (1.257 GHz)\n"
        f"{product_name}\nPolarization: {polarization}  |  "
        f"RFI Score: {results['score']:.1f}/100 ({results['severity']})",
        fontsize=13, fontweight="bold", y=0.98,
    )
    az = results["azimuth"]
    br = results["bright"]
    st = results["streaks"]

    finite_vals = data_db[np.isfinite(data_db)]
    vmin, vmax = (np.percentile(finite_vals, [2, 98]) if len(finite_vals) > 0 else (-30, 0))

    ds = max(1, max(data_db.shape) // 4096)

    # Panel 1: SAR image
    ax = axes[0, 0]
    ax.imshow(data_db[::ds, ::ds], cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title("SAR Backscatter (dB) — L-band")
    ax.set_xlabel("Range"); ax.set_ylabel("Azimuth")

    # Panel 2: Azimuth profile
    ax = axes[0, 1]
    rows = np.arange(len(az["row_means"]))
    ax.plot(az["row_means"], rows, linewidth=0.3, color="steelblue", label="Row mean (dB)")
    rfi_rows = rows[az["rfi_line_mask"]]
    if len(rfi_rows) > 0:
        ax.scatter(az["row_means"][az["rfi_line_mask"]], rfi_rows, c="red", s=1, zorder=5,
                   label=f"RFI lines ({az['n_rfi_lines']})")
    ax.axvline(az["global_median"], color="green", linestyle="--", linewidth=0.8,
               label=f"Median: {az['global_median']:.1f} dB")
    thresh = az["global_median"] + RFI_AZIMUTH_ZSCORE * az["robust_std"]
    ax.axvline(thresh, color="red", linestyle="--", linewidth=0.8,
               label=f"Threshold: {thresh:.1f} dB")
    ax.set_xlabel("Mean Backscatter (dB)"); ax.set_ylabel("Azimuth Line")
    ax.set_title("Azimuth-Line Mean Profile"); ax.invert_yaxis()
    ax.legend(fontsize=7, loc="lower right")

    # Panel 3: Bright pixel mask
    ax = axes[0, 2]
    db_small = data_db[::ds, ::ds]
    mask_small = br["bright_mask"][::ds, ::ds]
    rfi_overlay = np.zeros((*db_small.shape, 4))
    rfi_overlay[mask_small, :] = [1, 0, 0, 0.8]
    ax.imshow(db_small, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    ax.imshow(rfi_overlay, aspect="auto")
    ax.set_title(f"RFI Bright Pixels (red)\n{br['n_bright_pixels']} flagged")
    ax.set_xlabel("Range"); ax.set_ylabel("Azimuth")

    # Panel 4: Residual
    ax = axes[1, 0]
    residual = br["residual"]
    im = ax.imshow(residual[::ds, ::ds], cmap="RdBu_r", vmin=-10, vmax=10, aspect="auto")
    plt.colorbar(im, ax=ax, label="Residual (dB)")
    ax.set_title("Backscatter Residual\n(Data - Local Median Baseline)")
    ax.set_xlabel("Range"); ax.set_ylabel("Azimuth")

    # Panel 5: Z-score
    ax = axes[1, 1]
    ax.plot(az["zscores"], rows, linewidth=0.3, color="steelblue")
    ax.axvline(RFI_AZIMUTH_ZSCORE, color="red", linestyle="--",
               label=f"Z-score threshold ({RFI_AZIMUTH_ZSCORE})")
    ax.fill_betweenx(rows, RFI_AZIMUTH_ZSCORE, az["zscores"],
                      where=az["zscores"] > RFI_AZIMUTH_ZSCORE, color="red", alpha=0.3)
    ax.set_xlabel("Z-Score"); ax.set_ylabel("Azimuth Line")
    ax.set_title("Azimuth-Line Z-Score Profile"); ax.invert_yaxis()
    ax.legend(fontsize=8)

    # Panel 6: Streaks
    ax = axes[1, 2]
    ax.imshow(data_db[::ds, ::ds], cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    for s in st["streaks"]:
        rect = plt.Rectangle(
            (s["col_start"] / ds, (s["row_center"] - s["height"] / 2) / ds),
            s["width"] / ds, max(s["height"] / ds, 3),
            linewidth=1.5, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
    ax.set_title(f"Detected RFI Streaks ({st['n_streaks']} found)")
    ax.set_xlabel("Range"); ax.set_ylabel("Azimuth")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Report saved: {output_path}")


def print_summary(all_results: list):
    print("\n" + "=" * 72)
    print("  NISAR L-BAND RFI / GPS JAMMING DETECTION SUMMARY")
    print("  (L-band 1.257 GHz — ~30 MHz from GPS L2)")
    print("=" * 72)
    for r in all_results:
        print(f"\n  Product: {r['product_name']}")
        print(f"  Date:    {r['date']}")
        print(f"  Pol:     {r['polarization']}")
        print(f"  -----------------------------------")
        print(f"  RFI Score:       {r['score']:.1f} / 100")
        print(f"  Severity:        {r['severity']}")
        print(f"  Anomalous lines: {r['n_rfi_lines']} ({r['pct_rfi_lines']:.1f}%)")
        print(f"  Bright pixels:   {r['n_bright_pixels']} ({r['pct_bright']:.2f}%)")
        print(f"  Spectral peaks:  {r['spectral_peaks']}")
        print(f"  Linear streaks:  {r['n_streaks']}")
    print("\n" + "-" * 72)
    severities = [r["severity"] for r in all_results]
    if any(s == "HIGH" for s in severities):
        print("  !! HIGH RFI activity detected in L-band — strong GPS jamming indicator")
        print("     L-band sensitivity to GPS interference is ~100x greater than C-band.")
    elif any(s == "MODERATE" for s in severities):
        print("  !  MODERATE L-band RFI detected — possible GPS jamming")
    elif any(s == "LOW" for s in severities):
        print("  i  LOW L-band RFI detected — minor interference present")
    else:
        print("  ok Minimal/no RFI detected in L-band imagery.")
    print("-" * 72 + "\n")


# ---------------------------------------------------------------------------
# 6. Main Pipeline
# ---------------------------------------------------------------------------

def process_h5_file(h5_path: Path, output_dir: Path, subsample: int = 4) -> list:
    """Process a single NISAR RSLC HDF5 file through the RFI pipeline."""
    import h5py

    log.info(f"\n{'='*60}")
    log.info(f"Processing: {h5_path.name}")
    log.info(f"{'='*60}")

    # Explore structure on first run
    datasets = find_slc_datasets(h5_path)
    if not datasets:
        log.warning("No SLC datasets found. Exploring HDF5 structure...")
        explore_h5_structure(h5_path)
        return []

    product_name = h5_path.stem
    results = []

    for ds_info in datasets:
        pol = ds_info["polarization"]
        freq = ds_info["frequency"]

        log.info(f"\n  Processing {freq}/{pol}  shape={ds_info['shape']}")

        intensity, meta, full_shape = load_nisar_slc(h5_path, ds_info["path"], subsample=subsample)

        if intensity.size == 0 or np.all(intensity == 0):
            log.warning(f"  Empty data for {pol}, skipping")
            continue

        rfi = run_rfi_detection(intensity)

        # Extract date from metadata or filename
        date_str = "unknown"
        if "zeroDopplerStartTime" in meta:
            t = str(meta["zeroDopplerStartTime"])
            if len(t) >= 10:
                date_str = t[:10]
        if date_str == "unknown":
            # Try parsing from filename: ..._20260120T015447_...
            for part in product_name.split("_"):
                if len(part) >= 8 and part[:8].isdigit():
                    date_str = f"{part[:4]}-{part[4:6]}-{part[6:8]}"
                    break

        fig_name = f"nisar_rfi_{product_name}_{freq}_{pol}.png"
        fig_path = output_dir / fig_name
        plot_rfi_report(rfi["data_db"], rfi, product_name, pol, fig_path)

        results.append({
            "product_name": product_name,
            "sensor": "NISAR",
            "band": "L-band (1.257 GHz)",
            "polarization": pol,
            "frequency_group": freq,
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
            "metadata": {k: v for k, v in meta.items()
                         if k not in ("lat_grid", "lon_grid")},
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Detect GPS jamming (RFI) in NISAR L-band SAR imagery"
    )
    parser.add_argument(
        "--search-only", action="store_true",
        help="Only search the catalog; do not download or process",
    )
    parser.add_argument(
        "--local", type=str, nargs="+",
        help="Process local NISAR HDF5 file(s) instead of downloading",
    )
    parser.add_argument(
        "--start-date", default="2025-12-01",
        help="Start date (YYYY-MM-DD). Default: 2025-12-01",
    )
    parser.add_argument(
        "--end-date", default="2026-03-05",
        help="End date (YYYY-MM-DD). Default: 2026-03-05",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Output directory. Default: output/",
    )
    parser.add_argument(
        "--subsample", type=int, default=4,
        help="Subsample factor for SLC data (default: 4, reduces 14GB to ~1GB in memory)",
    )
    parser.add_argument(
        "--explore", action="store_true",
        help="Just explore HDF5 structure of --local files (no RFI detection)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Mode: Explore HDF5 structure ---
    if args.explore and args.local:
        for path_str in args.local:
            explore_h5_structure(Path(path_str))
        return

    # --- Mode: Process local files ---
    if args.local:
        all_results = []
        for path_str in args.local:
            h5_path = Path(path_str)
            if not h5_path.exists():
                log.error(f"File not found: {h5_path}")
                continue
            results = process_h5_file(h5_path, output_dir, subsample=args.subsample)
            all_results.extend(results)

        if all_results:
            print_summary(all_results)
            report_path = output_dir / "nisar_rfi_report.json"
            with open(report_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            log.info(f"JSON report saved: {report_path}")
        return

    # --- Mode: Search (and optionally download + process) ---
    products = search_nisar_products(args.start_date, args.end_date, TEHRAN_BBOX)

    if not products:
        log.warning("No NISAR RSLC products found for the specified dates and area.")
        return

    # Save search results
    search_results = [{
        "name": p["name"],
        "size_mb": p["size_mb"],
        "data_links": p["data_links"],
    } for p in products]
    search_path = output_dir / "nisar_search_results.json"
    with open(search_path, "w") as f:
        json.dump(search_results, f, indent=2)
    log.info(f"Search results saved: {search_path}")

    if args.search_only:
        print(f"\n  Search complete. Found {len(products)} NISAR RSLC products over Tehran.")
        print(f"  Results: {search_path}")
        print(f"  Run without --search-only to download and analyze.")
        return

    # --- Download and process ---
    download_dir = output_dir / "nisar_downloads"
    h5_paths = download_nisar_products(products, download_dir)

    all_results = []
    for h5_path in h5_paths:
        results = process_h5_file(h5_path, output_dir, subsample=args.subsample)
        all_results.extend(results)

    if all_results:
        print_summary(all_results)
        report_path = output_dir / "nisar_rfi_report.json"
        with open(report_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        log.info(f"JSON report saved: {report_path}")


if __name__ == "__main__":
    main()
