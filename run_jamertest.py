#!/usr/bin/env python3
"""
Run RFI detection on Jamertest Norway scenes with subsampling to manage memory.
Processes each scene sequentially, one TIFF at a time, with 4x downsampling.
"""
import json
import gc
import logging
from pathlib import Path
import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SUBSAMPLE = 4  # 4x downsample to reduce memory ~16x

SCENES = [
    ("S1A_IW_GRDH_1SDV_20250910T161557_20250910T161622_060928_0796C3_E727.SAFE",
     "2025-09-10", "S1A", "DESC", "Pre-event baseline (18:15 local)"),
    ("S1C_IW_GRDH_1SDV_20250911T160700_20250911T160725_004079_0081B2_D4BC.SAFE",
     "2025-09-11", "S1C", "DESC", "Day 1 Jamertest, DURING jamming (18:07 local)"),
    ("S1A_IW_GRDH_1SDV_20250916T054528_20250916T054553_061009_0799FD_4FA1.SAFE",
     "2025-09-16", "S1A", "ASC", "Mid-event, OUTSIDE jamming hours (07:45 local)"),
    ("S1C_IW_GRDH_1SDV_20250916T161506_20250916T161531_004152_0083F2_C908.SAFE",
     "2025-09-16", "S1C", "DESC", "Mid-event, DURING jamming (18:15 local)"),
    ("S1A_IW_GRDH_1SDV_20250918T052855_20250918T052920_061038_079B2F_ED03.SAFE",
     "2025-09-18", "S1A", "ASC", "Late event, OUTSIDE jamming hours (07:28 local)"),
    ("S1A_IW_GRDH_1SDV_20250920T163219_20250920T163244_061074_079C9A_A1A1.SAFE",
     "2025-09-20", "S1A", "DESC", "Post-event baseline (18:32 local)"),
]

BASE_DIR = Path(__file__).parent
DOWNLOAD_DIR = BASE_DIR / "output" / "downloads"
OUTPUT_DIR = BASE_DIR / "output" / "jamertest"


def intensity_to_db(data):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.where(data > 0, data, np.nan))


def detect_rfi_azimuth_lines(data_db):
    row_means = np.nanmean(data_db, axis=1)
    valid = np.isfinite(row_means)
    if not np.any(valid):
        return {"n_rfi_lines": 0, "pct_rfi_lines": 0.0, "total_lines": len(row_means)}
    med = np.nanmedian(row_means[valid])
    mad = np.nanmedian(np.abs(row_means[valid] - med))
    std_est = mad * 1.4826
    threshold = med + 3.0 * std_est
    rfi_mask = valid & (row_means > threshold)
    n_rfi = int(np.sum(rfi_mask))
    return {
        "n_rfi_lines": n_rfi,
        "pct_rfi_lines": round(100.0 * n_rfi / len(row_means), 2),
        "total_lines": len(row_means),
    }


def detect_rfi_bright_pixels(data_db):
    valid = np.isfinite(data_db)
    vals = data_db[valid]
    if len(vals) == 0:
        return {"n_bright_pixels": 0, "pct_bright": 0.0}
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    std_est = mad * 1.4826
    threshold = med + 4.0 * std_est
    bright_mask = valid & (data_db > threshold)
    n_bright = int(np.sum(bright_mask))
    total = int(np.sum(valid))
    return {
        "n_bright_pixels": n_bright,
        "pct_bright": round(100.0 * n_bright / total, 4) if total > 0 else 0.0,
    }


def detect_rfi_spectral(data_db, n_sample_cols=64):
    nrows, ncols = data_db.shape
    if ncols < n_sample_cols:
        return {"peak_counts": 0}
    col_indices = np.linspace(0, ncols - 1, n_sample_cols, dtype=int)
    peak_count = 0
    for ci in col_indices:
        col = data_db[:, ci]
        valid = np.isfinite(col)
        if np.sum(valid) < 100:
            continue
        col_v = col[valid]
        med = np.median(col_v)
        mad = np.median(np.abs(col_v - med))
        std_est = mad * 1.4826
        threshold = med + 5.0 * std_est
        peak_count += int(np.sum(col_v > threshold))
    return {"peak_counts": peak_count}


def compute_score(azimuth, bright, spectral):
    score = min(100.0, (
        azimuth["pct_rfi_lines"] * 2.0 +
        bright["pct_bright"] * 10.0 +
        min(spectral["peak_counts"], 100) * 0.3 +
        0  # skip streak detection to save memory
    ))
    if score > 60:
        severity = "HIGH"
    elif score > 30:
        severity = "MODERATE"
    elif score > 10:
        severity = "LOW"
    else:
        severity = "MINIMAL/NONE"
    return score, severity


def find_tifs(safe_dir):
    meas_dir = safe_dir / "measurement"
    if not meas_dir.exists():
        return []
    return sorted(meas_dir.glob("*.tiff")) + sorted(meas_dir.glob("*.tif"))


def process_one_tif(tif_path):
    log.info(f"  Loading {tif_path.name} (subsampled {SUBSAMPLE}x) ...")
    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        data = src.read(
            1,
            out_shape=(h // SUBSAMPLE, w // SUBSAMPLE),
            resampling=rasterio.enums.Resampling.average,
        ).astype(np.float32)
    log.info(f"    Shape: {data.shape}")

    data_db = intensity_to_db(data)
    del data
    gc.collect()

    azimuth = detect_rfi_azimuth_lines(data_db)
    bright = detect_rfi_bright_pixels(data_db)
    spectral = detect_rfi_spectral(data_db)
    del data_db
    gc.collect()

    score, severity = compute_score(azimuth, bright, spectral)
    return azimuth, bright, spectral, score, severity


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for safe_name, date, sat, direction, note in SCENES:
        safe_dir = DOWNLOAD_DIR / safe_name
        if not safe_dir.exists():
            log.warning(f"Missing: {safe_dir}")
            continue

        tifs = find_tifs(safe_dir)
        log.info(f"\n{'='*70}")
        log.info(f"Scene: {safe_name}")
        log.info(f"  Date: {date}  Sat: {sat}  Dir: {direction}")
        log.info(f"  Context: {note}")
        log.info(f"  Found {len(tifs)} measurement TIFFs")

        for tif_path in tifs:
            fname = tif_path.stem.lower()
            pol = "VH" if "vh" in fname else "VV" if "vv" in fname else "?"

            log.info(f"\n  Processing {pol} channel...")
            azimuth, bright, spectral, score, severity = process_one_tif(tif_path)

            log.info(f"    Azimuth lines: {azimuth['pct_rfi_lines']:.1f}% flagged")
            log.info(f"    Bright pixels: {bright['pct_bright']:.3f}%")
            log.info(f"    Spectral peaks: {spectral['peak_counts']}")
            log.info(f"    RFI Score: {score:.1f}/100 ({severity})")

            all_results.append({
                "product": safe_name.replace(".SAFE", ""),
                "date": date,
                "satellite": sat,
                "direction": direction,
                "polarization": pol,
                "note": note,
                "score": round(score, 1),
                "severity": severity,
                "pct_rfi_lines": azimuth["pct_rfi_lines"],
                "pct_bright": bright["pct_bright"],
                "spectral_peaks": spectral["peak_counts"],
                "n_rfi_lines": azimuth["n_rfi_lines"],
            })
            gc.collect()

    # Save results
    report_path = OUTPUT_DIR / "jamertest_rfi_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nResults saved to {report_path}")

    # Print summary table
    print("\n" + "="*90)
    print("JAMERTEST 2025 RFI DETECTION RESULTS")
    print("="*90)
    print(f"{'Date':<12} {'Sat':<5} {'Dir':<5} {'Pol':<4} {'Score':>6} {'Severity':<12} {'Context'}")
    print("-"*90)
    for r in all_results:
        print(f"{r['date']:<12} {r['satellite']:<5} {r['direction']:<5} {r['polarization']:<4} "
              f"{r['score']:>5.1f}  {r['severity']:<12} {r['note']}")
    print("="*90)


if __name__ == "__main__":
    main()
