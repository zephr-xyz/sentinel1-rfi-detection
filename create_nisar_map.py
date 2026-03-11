#!/usr/bin/env python3
"""
Create an interactive Leaflet map comparing NISAR L-band and Sentinel-1 C-band
RFI/GPS jamming detection over Tehran.

Produces output/nisar_rfi_map.html — a self-contained HTML file showing:
  - NISAR L-band SAR overlays + RFI polygons (Dec 27 baseline, Jan 8, Jan 20)
  - Sentinel-1 C-band overlays + RFI polygons (Feb 18-19, closest Tehran passes)
  - Layer toggle, legend, and date comparison
"""

import base64
import io
import json
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy import ndimage, interpolate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
NISAR_DIR = OUTPUT_DIR / "nisar_downloads"
S1_DIR = OUTPUT_DIR / "downloads"

RFI_AZIMUTH_ZSCORE = 3.0
RFI_PIXEL_ZSCORE = 4.0
SUBSAMPLE = 16
SAR_OVERLAY_WIDTH = 1024
RFI_MASK_WIDTH = 512


# ---------------------------------------------------------------------------
# NISAR HDF5 loading
# ---------------------------------------------------------------------------

def load_nisar_slc_subsampled(h5_path: Path, freq: str = "frequencyA", pol: str = "HH",
                               subsample: int = SUBSAMPLE) -> tuple:
    """Load NISAR RSLC, convert complex to intensity dB, return subsampled array + metadata."""
    ds_path = f"/science/LSAR/RSLC/swaths/{freq}/{pol}"
    log.info(f"  Loading {h5_path.name} [{freq}/{pol}] ...")

    with h5py.File(h5_path, "r") as f:
        ds = f[ds_path]
        full_h, full_w = ds.shape
        slc = ds[::subsample, ::subsample]

        # Get geolocation grid
        meta = {}
        grid_base = "/science/LSAR/RSLC/metadata/geolocationGrid"
        if grid_base in f:
            ggrp = f[grid_base]
            for key in ["heightAboveEllipsoid", "zeroDopplerTime",
                         "slantRange", "coordinateX", "coordinateY",
                         "latitude", "longitude"]:
                if key in ggrp:
                    meta[key] = ggrp[key][()]

        # Identification metadata
        id_base = "/science/LSAR/identification"
        if id_base in f:
            igrp = f[id_base]
            for key in ["boundingPolygon", "zeroDopplerStartTime", "zeroDopplerEndTime",
                         "orbitPassDirection", "trackNumber"]:
                if key in igrp:
                    val = igrp[key][()]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8")
                    meta[key] = val

    intensity = np.abs(slc).astype(np.float64) ** 2
    sub_h, sub_w = intensity.shape
    log.info(f"    Shape: {full_h}x{full_w} -> {sub_h}x{sub_w}")
    return intensity, meta, full_h, full_w


def get_nisar_bounds(meta: dict) -> tuple:
    """Extract lat/lon bounds from NISAR metadata."""
    if "latitude" in meta and "longitude" in meta:
        lats = meta["latitude"]
        lons = meta["longitude"]
        return (float(np.nanmin(lons)), float(np.nanmin(lats)),
                float(np.nanmax(lons)), float(np.nanmax(lats)))
    if "boundingPolygon" in meta:
        bp = meta["boundingPolygon"]
        if isinstance(bp, str):
            # Parse WKT-like polygon
            import re
            coords = re.findall(r'([-\d.]+)\s+([-\d.]+)', bp)
            if coords:
                lons = [float(c[0]) for c in coords]
                lats = [float(c[1]) for c in coords]
                return (min(lons), min(lats), max(lons), max(lats))
    return None


def nisar_intensity_to_overlay(intensity: np.ndarray, bounds: tuple,
                                out_width: int = SAR_OVERLAY_WIDTH) -> tuple:
    """
    Convert NISAR intensity to a geographic overlay PNG.

    Since NISAR RSLC is in radar coordinates (not geocoded), we use the
    bounding box as an approximation for the image extent.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        data_db = 10.0 * np.log10(np.where(intensity > 0, intensity, np.nan))

    h, w = data_db.shape
    aspect = h / max(w, 1)
    out_height = int(out_width * aspect)

    # Resize to output dimensions
    from scipy.ndimage import zoom
    scale_h = out_height / h
    scale_w = out_width / w
    # Replace NaN before zoom
    db_filled = np.where(np.isfinite(data_db), data_db, np.nanmedian(data_db))
    resized = zoom(db_filled, (scale_h, scale_w), order=1)

    valid_vals = resized[np.isfinite(resized)]
    if len(valid_vals) == 0:
        return "", bounds
    p2, p98 = np.percentile(valid_vals, [2, 98])
    stretched = np.clip((resized - p2) / max(p98 - p2, 0.01), 0, 1)
    gray = (stretched * 255).astype(np.uint8)

    # Build RGBA
    rgba = np.zeros((*resized.shape, 4), dtype=np.uint8)
    rgba[:, :, 0] = gray
    rgba[:, :, 1] = gray
    rgba[:, :, 2] = gray
    rgba[:, :, 3] = 180  # semi-transparent

    img = Image.fromarray(rgba)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    log.info(f"    Overlay PNG: {len(buf.getvalue())/1024:.0f} KB")
    return b64, bounds


# ---------------------------------------------------------------------------
# Sentinel-1 loading (reuse from create_map.py)
# ---------------------------------------------------------------------------

def parse_geolocation_grid(annotation_xml: Path) -> dict:
    tree = ET.parse(annotation_xml)
    root = tree.getroot()
    points = root.findall(".//geolocationGridPoint")
    lines = np.array([int(p.find("line").text) for p in points])
    pixels = np.array([int(p.find("pixel").text) for p in points])
    lats = np.array([float(p.find("latitude").text) for p in points])
    lons = np.array([float(p.find("longitude").text) for p in points])
    unique_lines = np.unique(lines)
    unique_pixels = np.unique(pixels)
    n_lines = len(unique_lines)
    n_pixels = len(unique_pixels)
    lat_grid = lats.reshape(n_lines, n_pixels)
    lon_grid = lons.reshape(n_lines, n_pixels)
    return {
        "unique_lines": unique_lines, "unique_pixels": unique_pixels,
        "lat_grid": lat_grid, "lon_grid": lon_grid,
        "lat_min": float(lats.min()), "lat_max": float(lats.max()),
        "lon_min": float(lons.min()), "lon_max": float(lons.max()),
    }


def load_s1_subsampled(tif_path: Path, subsample: int = SUBSAMPLE) -> tuple:
    import rasterio
    with rasterio.open(tif_path) as src:
        full_h, full_w = src.height, src.width
        out_h = full_h // subsample
        out_w = full_w // subsample
        data = src.read(1, out_shape=(out_h, out_w)).astype(np.float64)
    log.info(f"  Loaded {tif_path.name}: {full_h}x{full_w} -> {data.shape}")
    return data, full_h, full_w


def warp_to_geographic(sensor_data, grid, full_h, full_w, out_width, order=1):
    sub_h, sub_w = sensor_data.shape
    lat_min, lat_max = grid["lat_min"], grid["lat_max"]
    lon_min, lon_max = grid["lon_min"], grid["lon_max"]
    lat_pad = (lat_max - lat_min) * 0.01
    lon_pad = (lon_max - lon_min) * 0.01
    lat_min -= lat_pad; lat_max += lat_pad
    lon_min -= lon_pad; lon_max += lon_pad
    aspect = (lat_max - lat_min) / (lon_max - lon_min)
    out_height = int(out_width * aspect)
    ul = grid["unique_lines"]
    up = grid["unique_pixels"]
    line_grid_2d, pixel_grid_2d = np.meshgrid(ul, up, indexing="ij")
    known_lats = grid["lat_grid"].ravel()
    known_lons = grid["lon_grid"].ravel()
    known_lines = line_grid_2d.ravel()
    known_pixels = pixel_grid_2d.ravel()
    known_points = np.column_stack([known_lons, known_lats])
    dest_lats = np.linspace(lat_max, lat_min, out_height)
    dest_lons = np.linspace(lon_min, lon_max, out_width)
    dest_lon_grid, dest_lat_grid = np.meshgrid(dest_lons, dest_lats)
    dest_points = np.column_stack([dest_lon_grid.ravel(), dest_lat_grid.ravel()])
    inv_lines = interpolate.griddata(known_points, known_lines, dest_points, method="cubic", fill_value=-1).reshape(out_height, out_width)
    inv_pixels = interpolate.griddata(known_points, known_pixels, dest_points, method="cubic", fill_value=-1).reshape(out_height, out_width)
    scale_line = sub_h / full_h
    scale_pixel = sub_w / full_w
    valid = (inv_lines >= 0) & (inv_lines < full_h) & (inv_pixels >= 0) & (inv_pixels < full_w)
    coords = np.array([inv_lines * scale_line, inv_pixels * scale_pixel])
    coords[0, ~valid] = 0; coords[1, ~valid] = 0
    data_for_interp = sensor_data.copy()
    nan_mask = ~np.isfinite(data_for_interp)
    if np.any(nan_mask):
        data_for_interp[nan_mask] = 0
    warped = ndimage.map_coordinates(data_for_interp, coords, order=order, mode="constant", cval=0)
    warped[~valid] = np.nan
    bounds = (lon_min, lat_min, lon_max, lat_max)
    return warped, bounds, valid


def sar_to_base64_png(warped_db, valid_mask, alpha_inside=180):
    h, w = warped_db.shape
    valid_vals = warped_db[valid_mask & np.isfinite(warped_db)]
    if len(valid_vals) == 0:
        return ""
    p2, p98 = np.percentile(valid_vals, [2, 98])
    stretched = np.clip((warped_db - p2) / max(p98 - p2, 0.01), 0, 1)
    np.nan_to_num(stretched, copy=False, nan=0.0)
    gray = (stretched * 255).astype(np.uint8)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = gray; rgba[:, :, 1] = gray; rgba[:, :, 2] = gray
    rgba[:, :, 3] = np.where(valid_mask, alpha_inside, 0).astype(np.uint8)
    img = Image.fromarray(rgba)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    log.info(f"    PNG: {len(buf.getvalue())/1024:.0f} KB")
    return b64


# ---------------------------------------------------------------------------
# RFI detection (lightweight for map)
# ---------------------------------------------------------------------------

def intensity_to_db(data):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.where(data > 0, data, np.nan))


def detect_bright_pixels(data_db):
    from scipy.ndimage import zoom as scipy_zoom
    h, w = data_db.shape
    scale = max(4, min(h, w) // 256)
    small = data_db[::scale, ::scale].copy()
    nan_mask = ~np.isfinite(small)
    fill_val = np.nanmedian(small) if np.any(~nan_mask) else 0.0
    small[nan_mask] = fill_val
    baseline_small = ndimage.median_filter(small, size=15)
    baseline = scipy_zoom(baseline_small, (h / baseline_small.shape[0], w / baseline_small.shape[1]))[:h, :w]
    baseline[~np.isfinite(data_db)] = np.nan
    residual = data_db - baseline
    valid = residual[np.isfinite(residual)]
    if len(valid) > 0:
        med = np.median(valid)
        mad = np.median(np.abs(valid - med))
        rstd = max(1.4826 * mad, 0.01)
        bright = np.isfinite(residual) & (residual > (med + RFI_PIXEL_ZSCORE * rstd))
    else:
        bright = np.zeros_like(data_db, dtype=bool)
    n = int(np.sum(bright))
    log.info(f"    Bright pixels: {n} ({n / max(np.sum(np.isfinite(data_db)), 1) * 100:.2f}%)")
    return bright


def compute_rfi_score(data_db, bright_mask):
    """Compute simplified RFI score."""
    from scipy import signal
    # Azimuth lines
    row_means = np.nanmean(data_db, axis=1)
    gmed = np.nanmedian(row_means)
    mad = np.nanmedian(np.abs(row_means - gmed))
    rstd = max(1.4826 * mad, 0.01)
    pct_rfi_lines = np.sum((row_means - gmed) / rstd > RFI_AZIMUTH_ZSCORE) / len(row_means) * 100

    # Bright pixel pct
    pct_bright = np.sum(bright_mask) / max(np.sum(np.isfinite(data_db)), 1) * 100

    # Spectral peaks (sample 64 columns)
    h, w = data_db.shape
    col_indices = np.linspace(0, w - 1, 64, dtype=int)
    peak_counts = 0
    for ci in col_indices:
        col = data_db[:, ci]
        v = col[np.isfinite(col)]
        if len(v) < 256:
            continue
        v = v - np.mean(v)
        win = signal.windows.hann(len(v))
        spec = np.abs(np.fft.rfft(v * win))
        spec_db = 20 * np.log10(spec + 1e-30)
        floor = np.median(spec_db)
        above = spec_db > (floor + 10.0)
        above[:5] = False
        peak_counts += int(np.sum(above))

    score = min(100.0, pct_rfi_lines * 2 + pct_bright * 10 + min(peak_counts, 100) * 0.3)
    if score > 60: sev = "HIGH"
    elif score > 30: sev = "MODERATE"
    elif score > 10: sev = "LOW"
    else: sev = "MINIMAL/NONE"
    log.info(f"    RFI score: {score:.1f}/100 ({sev})")
    return score, sev, peak_counts


# ---------------------------------------------------------------------------
# GeoJSON vectorization
# ---------------------------------------------------------------------------

def vectorize_rfi(bright_mask, bounds, valid_mask, props, min_area=1e-4):
    from rasterio.transform import from_bounds
    from rasterio.features import shapes
    from shapely.geometry import shape, mapping
    from shapely.validation import make_valid
    lon_min, lat_min, lon_max, lat_max = bounds
    h, w = bright_mask.shape
    mask = bright_mask & valid_mask
    mask = ndimage.binary_dilation(mask, iterations=3)
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)
    features = []
    for geom, val in shapes(mask.astype(np.uint8), mask=(mask.astype(np.uint8) == 1), transform=transform):
        if val == 0:
            continue
        poly = make_valid(shape(geom))
        if poly.is_empty or poly.area < min_area:
            continue
        poly = poly.simplify(0.005, preserve_topology=True)
        if poly.is_empty:
            continue
        features.append({"type": "Feature", "geometry": mapping(poly), "properties": props})
    log.info(f"    Vectorized {len(features)} RFI polygons")
    return features


# ---------------------------------------------------------------------------
# HTML map builder
# ---------------------------------------------------------------------------

def build_html(layers: list) -> str:
    overlay_js = []
    geojson_js = []
    overlay_names = {}
    default_on = []

    for i, ly in enumerate(layers):
        lbl = ly["label"]
        b = ly["bounds"]
        bounds_js = f"[[{b[1]:.6f}, {b[0]:.6f}], [{b[3]:.6f}, {b[2]:.6f}]]"

        sar_var = f"sar_{i}"
        overlay_js.append(
            f'var {sar_var} = L.imageOverlay('
            f'"data:image/png;base64,{ly["b64_png"]}", {bounds_js});'
        )
        overlay_names[f"SAR: {lbl}"] = sar_var

        if ly.get("geojson_features"):
            gj_var = f"rfi_{i}"
            gj_data = json.dumps({"type": "FeatureCollection", "features": ly["geojson_features"]})
            geojson_js.append(
                f'var {gj_var} = L.geoJSON({gj_data}, {{\n'
                f'  style: function(f) {{\n'
                f'    var s = f.properties.severity;\n'
                f'    var c = s==="HIGH"?"#ff0000":s==="MODERATE"?"#ff8c00":"#ffd700";\n'
                f'    return {{color:c,weight:1.5,fillColor:c,fillOpacity:0.3,opacity:0.8}};\n'
                f'  }},\n'
                f'  onEachFeature: function(f,l) {{\n'
                f'    var p=f.properties;\n'
                f'    l.bindPopup("<b>RFI Detection</b><br>"\n'
                f'      +"Sensor: "+p.sensor+"<br>"\n'
                f'      +"Band: "+p.band+"<br>"\n'
                f'      +"Date: "+p.date+"<br>"\n'
                f'      +"Pol: "+p.polarization+"<br>"\n'
                f'      +"Score: "+p.rfi_score+"/100<br>"\n'
                f'      +"Severity: <b>"+p.severity+"</b><br>"\n'
                f'      +"Spectral peaks: "+p.spectral_peaks);\n'
                f'  }}\n'
                f'}});'
            )
            overlay_names[f"RFI: {lbl}"] = gj_var

        if ly.get("default_on"):
            default_on.append(sar_var)
            if ly.get("geojson_features"):
                default_on.append(gj_var)

    overlays_entries = [f'    "{k}": {v}' for k, v in overlay_names.items()]
    overlays_obj = "{\n" + ",\n".join(overlays_entries) + "\n  }"
    default_add = "\n  ".join(f"{v}.addTo(map);" for v in default_on)

    all_lats, all_lons = [], []
    for ly in layers:
        b = ly["bounds"]
        all_lats.extend([b[1], b[3]])
        all_lons.extend([b[0], b[2]])

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>NISAR L-band vs Sentinel-1 C-band RFI Detection</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body {{ margin: 0; padding: 0; }}
    #map {{ width: 100%; height: 100vh; }}
    .legend {{
      background: white; padding: 12px 16px; border-radius: 5px;
      box-shadow: 0 0 15px rgba(0,0,0,0.2); font: 13px/1.6 Arial, sans-serif;
      max-width: 280px;
    }}
    .legend h4 {{ margin: 0 0 8px 0; font-size: 14px; }}
    .legend-item {{ display: flex; align-items: center; margin-bottom: 4px; }}
    .legend-color {{ width: 20px; height: 14px; margin-right: 8px; border: 1px solid #666; flex-shrink: 0; }}
    .info-box {{
      background: white; padding: 12px 16px; border-radius: 5px;
      box-shadow: 0 0 15px rgba(0,0,0,0.2); font: 12px/1.5 Arial, sans-serif;
      max-width: 320px;
    }}
    .info-box h4 {{ margin: 0 0 6px 0; font-size: 13px; }}
    .info-box table {{ border-collapse: collapse; width: 100%; font-size: 11px; }}
    .info-box td, .info-box th {{ padding: 2px 6px; text-align: left; border-bottom: 1px solid #eee; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script>
  var map = L.map('map').fitBounds([[{min(all_lats):.4f}, {min(all_lons):.4f}], [{max(all_lats):.4f}, {max(all_lons):.4f}]]);

  var cartoVoyager = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution: '&copy; CARTO &copy; OSM', maxZoom: 19, subdomains: 'abcd'
  }}).addTo(map);

  var esriImagery = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
    attribution: '&copy; Esri', maxZoom: 19
  }});

  var baseMaps = {{"Map": cartoVoyager, "Satellite": esriImagery}};

  {"  ".join(overlay_js)}
  {"  ".join(geojson_js)}
  {default_add}

  L.control.layers(baseMaps, {overlays_obj}, {{collapsed: false}}).addTo(map);

  // Legend
  var legend = L.control({{position: 'bottomright'}});
  legend.onAdd = function() {{
    var div = L.DomUtil.create('div', 'legend');
    div.innerHTML = '<h4>RFI Severity</h4>'
      + '<div class="legend-item"><div class="legend-color" style="background:#ff0000;opacity:0.5;"></div>HIGH (&gt;60)</div>'
      + '<div class="legend-item"><div class="legend-color" style="background:#ff8c00;opacity:0.5;"></div>MODERATE (30-60)</div>'
      + '<div class="legend-item"><div class="legend-color" style="background:#ffd700;opacity:0.5;"></div>LOW (10-30)</div>'
      + '<div class="legend-item"><div class="legend-color" style="background:#888;opacity:0.3;"></div>SAR overlay</div>';
    return div;
  }};
  legend.addTo(map);

  // Info panel
  var info = L.control({{position: 'topright'}});
  info.onAdd = function() {{
    var div = L.DomUtil.create('div', 'info-box');
    div.innerHTML = '<h4>NISAR L-band vs Sentinel-1 C-band</h4>'
      + '<table>'
      + '<tr><th></th><th>NISAR</th><th>Sentinel-1</th></tr>'
      + '<tr><td>Band</td><td>L (1.26 GHz)</td><td>C (5.4 GHz)</td></tr>'
      + '<tr><td>GPS L2 offset</td><td><b>~30 MHz</b></td><td>~4.2 GHz</td></tr>'
      + '<tr><td>GPS sensitivity</td><td><b>Very High</b></td><td>Indirect</td></tr>'
      + '<tr><td>Resolution</td><td>3-10 m</td><td>10-20 m</td></tr>'
      + '<tr><td>Repeat</td><td>12 days</td><td>6 days</td></tr>'
      + '</table>'
      + '<p style="margin:6px 0 0;font-size:10px;color:#666;">GPS jammers emit broadband noise near L1/L2 (1.2-1.6 GHz). '
      + 'NISAR L-band receives this interference directly.</p>';
    return div;
  }};
  info.addTo(map);
  </script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def find_nisar_products():
    """Find downloaded NISAR HDF5 files, preferring frame 020 (Tehran coverage)."""
    if not NISAR_DIR.exists():
        return []
    all_files = sorted(NISAR_DIR.glob("NISAR_L1_PR_RSLC_*.h5"))

    # Group by cycle number, prefer frame 020 (Tehran) over 019
    # Filename: NISAR_L1_PR_RSLC_{cycle}_{track}_A_{frame}_{...}
    by_cycle = {}
    for f in all_files:
        parts = f.stem.split("_")
        # parts[4]=cycle (008/009/010), parts[7]=frame (019/020)
        cycle = parts[4] if len(parts) > 7 else "000"
        frame = parts[7] if len(parts) > 7 else "000"
        if cycle not in by_cycle or frame > by_cycle[cycle][1]:
            by_cycle[cycle] = (f, frame)

    result = [v[0] for v in sorted(by_cycle.values(), key=lambda x: x[0].name)]
    log.info(f"  Selected {len(result)} NISAR files (preferring frame 020):")
    for f in result:
        log.info(f"    {f.name}")
    return result


def find_s1_tehran_products():
    """Find Sentinel-1 products over Tehran (Feb 18-19 passes)."""
    if not S1_DIR.exists():
        return []
    products = []
    for safe_dir in sorted(S1_DIR.glob("S1A*20260218*.SAFE")) + sorted(S1_DIR.glob("S1A*20260219*.SAFE")):
        ann_dir = safe_dir / "annotation"
        meas_dir = safe_dir / "measurement"
        if not ann_dir.exists() or not meas_dir.exists():
            continue
        for pol in ["vv"]:  # Just VV for cleaner map
            ann_xmls = [f for f in ann_dir.glob("*.xml") if f.name.startswith("s1") and f"-{pol}-" in f.name]
            tifs = [f for f in meas_dir.glob("*.tiff") if f"-{pol}-" in f.name]
            if ann_xmls and tifs:
                products.append({
                    "safe_dir": safe_dir,
                    "annotation_xml": ann_xmls[0],
                    "tif_path": tifs[0],
                    "polarization": pol.upper(),
                    "product_name": safe_dir.stem,
                })
    return products


def process_nisar_layer(h5_path: Path) -> dict:
    """Process one NISAR file into a map layer."""
    name = h5_path.stem

    # Extract date
    date_str = "unknown"
    for part in name.split("_"):
        if len(part) >= 8 and part[:8].isdigit():
            date_str = f"{part[:4]}-{part[4:6]}-{part[6:8]}"
            break

    # Load L-band HH (most sensitive to GPS jamming)
    intensity, meta, full_h, full_w = load_nisar_slc_subsampled(h5_path, "frequencyA", "HH")

    bounds = get_nisar_bounds(meta)
    if bounds is None:
        # Fallback: use Tehran approximate bounds
        bounds = (48.0, 33.0, 52.5, 38.0)
        log.warning("  No geolocation found, using approximate bounds")

    data_db = intensity_to_db(intensity)
    bright = detect_bright_pixels(data_db)
    score, severity, peaks = compute_rfi_score(data_db, bright)

    b64, _ = nisar_intensity_to_overlay(intensity, bounds)

    # For RFI vectorization, use full image extent as valid
    valid = np.ones(bright.shape, dtype=bool)

    # Scale bright mask to match overlay dimensions if needed
    props = {
        "sensor": "NISAR", "band": "L-band (1.26 GHz)", "date": date_str,
        "polarization": "HH", "rfi_score": str(round(score, 1)),
        "severity": severity, "spectral_peaks": str(peaks),
    }

    # Simple bounding-box based vectorization
    from rasterio.transform import from_bounds
    from rasterio.features import shapes
    from shapely.geometry import shape, mapping
    from shapely.validation import make_valid
    h, w = bright.shape
    mask = ndimage.binary_dilation(bright, iterations=3)
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], w, h)
    features = []
    for geom, val in shapes(mask.astype(np.uint8), mask=(mask.astype(np.uint8) == 1), transform=transform):
        if val == 0:
            continue
        poly = make_valid(shape(geom))
        if poly.is_empty or poly.area < 1e-4:
            continue
        poly = poly.simplify(0.005, preserve_topology=True)
        if not poly.is_empty:
            features.append({"type": "Feature", "geometry": mapping(poly), "properties": props})
    log.info(f"    {len(features)} RFI polygons")

    is_jamming = date_str >= "2026-01-08"
    label = f"NISAR L-band {date_str} HH" + (" [jamming]" if is_jamming else " [baseline]")

    return {
        "label": label,
        "b64_png": b64,
        "bounds": bounds,
        "geojson_features": features,
        "score": score,
        "severity": severity,
        "default_on": date_str == "2026-01-20",
    }


def process_s1_layer(product: dict) -> dict:
    """Process one Sentinel-1 product into a map layer."""
    pname = product["product_name"]
    pol = product["polarization"]

    # Extract date
    parts = pname.split("_")
    date_str = "unknown"
    for p in parts:
        if len(p) >= 8 and p[:8].isdigit():
            date_str = f"{p[:4]}-{p[4:6]}-{p[6:8]}"
            break

    log.info(f"\nProcessing S1: {pname} {pol}")

    grid = parse_geolocation_grid(product["annotation_xml"])
    data, full_h, full_w = load_s1_subsampled(product["tif_path"])
    data_db = intensity_to_db(data)
    bright = detect_bright_pixels(data_db)
    score, severity, peaks = compute_rfi_score(data_db, bright)

    # Warp SAR overlay
    warped_db, bounds, valid = warp_to_geographic(data_db, grid, full_h, full_w, SAR_OVERLAY_WIDTH)
    b64 = sar_to_base64_png(warped_db, valid)

    # Warp and vectorize RFI
    bright_f = bright.astype(np.float64)
    warped_bright, rb, rv = warp_to_geographic(bright_f, grid, full_h, full_w, RFI_MASK_WIDTH, order=0)
    warped_bool = (warped_bright > 0.5) & rv

    props = {
        "sensor": "Sentinel-1", "band": "C-band (5.4 GHz)", "date": date_str,
        "polarization": pol, "rfi_score": str(round(score, 1)),
        "severity": severity, "spectral_peaks": str(peaks),
    }
    features = vectorize_rfi(warped_bool, rb, rv, props)

    label = f"S1 C-band {date_str} {pol}"
    return {
        "label": label,
        "b64_png": b64,
        "bounds": bounds,
        "geojson_features": features,
        "score": score,
        "severity": severity,
        "default_on": False,
    }


def main():
    log.info("=" * 60)
    log.info("Creating NISAR L-band vs Sentinel-1 C-band RFI map")
    log.info("=" * 60)

    layers = []

    # Process NISAR products
    nisar_files = find_nisar_products()
    log.info(f"Found {len(nisar_files)} NISAR product(s)")
    for h5_path in nisar_files:
        try:
            layer = process_nisar_layer(h5_path)
            layers.append(layer)
        except Exception as e:
            log.error(f"Error processing {h5_path.name}: {e}")
            import traceback; traceback.print_exc()

    # Process Sentinel-1 products
    s1_products = find_s1_tehran_products()
    log.info(f"Found {len(s1_products)} Sentinel-1 Tehran product(s)")
    for prod in s1_products:
        try:
            layer = process_s1_layer(prod)
            layers.append(layer)
        except Exception as e:
            log.error(f"Error processing {prod['product_name']}: {e}")
            import traceback; traceback.print_exc()

    if not layers:
        log.error("No layers produced!")
        sys.exit(1)

    # Sort layers by date
    layers.sort(key=lambda x: x["label"])

    log.info(f"\nBuilding HTML with {len(layers)} layers...")
    html = build_html(layers)

    out_path = OUTPUT_DIR / "nisar_rfi_map.html"
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    log.info(f"\nWrote {out_path} ({size_mb:.1f} MB)")
    log.info("Open in a browser to view.")


if __name__ == "__main__":
    main()
