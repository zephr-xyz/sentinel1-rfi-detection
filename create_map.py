#!/usr/bin/env python3
"""
Create an interactive Leaflet map showing Sentinel-1 RFI/GPS jamming detection.

Produces a single self-contained HTML file (output/rfi_map.html) that opens
directly in a browser with no server needed. Includes:
  - OSM basemap centered on Tehran
  - SAR image overlays (base64 PNG) for each product/polarization
  - RFI detection polygons (GeoJSON) styled by severity
  - Layer toggle and click popups with detection info
"""

import base64
import io
import json
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
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
DOWNLOADS_DIR = OUTPUT_DIR / "downloads"

# RFI detection thresholds (match sentinel1_rfi_demo.py)
RFI_AZIMUTH_ZSCORE = 3.0
RFI_PIXEL_ZSCORE = 4.0

# Subsample factor for reading TIFFs
SUBSAMPLE = 16

# Output image widths
SAR_OVERLAY_WIDTH = 1024
RFI_MASK_WIDTH = 512


# ---------------------------------------------------------------------------
# Step 1: Parse geolocation grids from annotation XMLs
# ---------------------------------------------------------------------------

def parse_geolocation_grid(annotation_xml: Path) -> dict:
    """Parse the 210-point geolocation grid from a Sentinel-1 annotation XML."""
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

    log.info(f"  Geolocation grid: {n_lines}x{n_pixels} = {len(points)} points")
    log.info(f"  Lat: {lats.min():.4f} - {lats.max():.4f}")
    log.info(f"  Lon: {lons.min():.4f} - {lons.max():.4f}")

    # Reshape into 2D grids (n_lines x n_pixels)
    lat_grid = lats.reshape(n_lines, n_pixels)
    lon_grid = lons.reshape(n_lines, n_pixels)

    return {
        "unique_lines": unique_lines,
        "unique_pixels": unique_pixels,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "lat_min": float(lats.min()),
        "lat_max": float(lats.max()),
        "lon_min": float(lons.min()),
        "lon_max": float(lons.max()),
    }


def build_forward_interpolators(grid: dict):
    """Build interpolators: (line, pixel) -> (lat, lon)."""
    ul = grid["unique_lines"]
    up = grid["unique_pixels"]
    # RectBivariateSpline needs monotonically increasing x, y
    lat_interp = interpolate.RectBivariateSpline(ul, up, grid["lat_grid"], kx=3, ky=3)
    lon_interp = interpolate.RectBivariateSpline(ul, up, grid["lon_grid"], kx=3, ky=3)
    return lat_interp, lon_interp


# ---------------------------------------------------------------------------
# Step 2: Load SAR images at reduced resolution
# ---------------------------------------------------------------------------

def load_sar_subsampled(tif_path: Path, subsample: int = SUBSAMPLE) -> np.ndarray:
    """Load a SAR GeoTIFF at reduced resolution using rasterio out_shape."""
    with rasterio.open(tif_path) as src:
        full_h, full_w = src.height, src.width
        out_h = full_h // subsample
        out_w = full_w // subsample
        data = src.read(1, out_shape=(out_h, out_w)).astype(np.float64)
    log.info(f"  Loaded {tif_path.name}: {full_h}x{full_w} -> {data.shape}")
    return data, full_h, full_w


# ---------------------------------------------------------------------------
# Step 3: Run RFI detection on subsampled data
# ---------------------------------------------------------------------------

def intensity_to_db(data: np.ndarray) -> np.ndarray:
    """Convert linear intensity to dB, handling zeros."""
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.where(data > 0, data, np.nan))
    return db


def detect_rfi_bright_pixels(data_db: np.ndarray) -> np.ndarray:
    """Detect anomalously bright pixels. Returns boolean bright_mask."""
    from scipy.ndimage import zoom as scipy_zoom
    h, w = data_db.shape
    scale = max(4, min(h, w) // 256)

    small = data_db[::scale, ::scale].copy()
    nan_mask_small = ~np.isfinite(small)
    fill_val = np.nanmedian(small) if np.any(~nan_mask_small) else 0.0
    small[nan_mask_small] = fill_val

    baseline_small = ndimage.median_filter(small, size=15)
    baseline = scipy_zoom(
        baseline_small,
        (h / baseline_small.shape[0], w / baseline_small.shape[1]),
    )
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
        bright_mask = np.isfinite(residual) & (
            residual > (med + RFI_PIXEL_ZSCORE * rstd)
        )
    else:
        bright_mask = np.zeros_like(data_db, dtype=bool)

    n_bright = int(np.sum(bright_mask))
    log.info(f"    Bright pixels: {n_bright} ({n_bright / max(np.sum(np.isfinite(data_db)), 1) * 100:.2f}%)")
    return bright_mask


# ---------------------------------------------------------------------------
# Step 4: Warp SAR images + RFI masks to EPSG:4326
# ---------------------------------------------------------------------------

def warp_to_geographic(
    sensor_data: np.ndarray,
    lat_interp,
    lon_interp,
    full_h: int,
    full_w: int,
    grid: dict,
    out_width: int,
    order: int = 1,
) -> tuple:
    """
    Warp sensor-coordinate data to a geographic (lat/lon) grid.

    Returns (warped_data, bounds) where bounds = (lon_min, lat_min, lon_max, lat_max).
    """
    sub_h, sub_w = sensor_data.shape
    lat_min, lat_max = grid["lat_min"], grid["lat_max"]
    lon_min, lon_max = grid["lon_min"], grid["lon_max"]

    # Add small buffer
    lat_pad = (lat_max - lat_min) * 0.01
    lon_pad = (lon_max - lon_min) * 0.01
    lat_min -= lat_pad
    lat_max += lat_pad
    lon_min -= lon_pad
    lon_max += lon_pad

    aspect = (lat_max - lat_min) / (lon_max - lon_min)
    out_height = int(out_width * aspect)

    # Create destination grid of (lat, lon) coordinates
    dest_lats = np.linspace(lat_max, lat_min, out_height)  # top to bottom
    dest_lons = np.linspace(lon_min, lon_max, out_width)
    dest_lon_grid, dest_lat_grid = np.meshgrid(dest_lons, dest_lats)

    # Build inverse mapping: (lat, lon) -> (line, pixel) using griddata
    # Use the GCP grid points
    ul = grid["unique_lines"]
    up = grid["unique_pixels"]
    # Create full grid of known (lat, lon) -> (line, pixel) pairs
    line_grid_2d, pixel_grid_2d = np.meshgrid(ul, up, indexing="ij")
    known_lats = grid["lat_grid"].ravel()
    known_lons = grid["lon_grid"].ravel()
    known_lines = line_grid_2d.ravel()
    known_pixels = pixel_grid_2d.ravel()

    known_points = np.column_stack([known_lons, known_lats])
    dest_points = np.column_stack([dest_lon_grid.ravel(), dest_lat_grid.ravel()])

    # Inverse map: geographic -> sensor coordinates
    inv_lines = interpolate.griddata(
        known_points, known_lines, dest_points, method="cubic", fill_value=-1
    ).reshape(out_height, out_width)
    inv_pixels = interpolate.griddata(
        known_points, known_pixels, dest_points, method="cubic", fill_value=-1
    ).reshape(out_height, out_width)

    # Scale to subsampled coordinates
    scale_line = sub_h / full_h
    scale_pixel = sub_w / full_w
    inv_lines_sub = inv_lines * scale_line
    inv_pixels_sub = inv_pixels * scale_pixel

    # Create validity mask (inside the SAR footprint)
    valid = (
        (inv_lines >= 0)
        & (inv_lines < full_h)
        & (inv_pixels >= 0)
        & (inv_pixels < full_w)
    )

    # Resample using map_coordinates
    coords = np.array([inv_lines_sub, inv_pixels_sub])
    # Replace invalid coords with 0 temporarily
    coords[0, ~valid] = 0
    coords[1, ~valid] = 0

    # Handle NaN in input data for map_coordinates
    data_for_interp = sensor_data.copy()
    nan_mask = ~np.isfinite(data_for_interp)
    if np.any(nan_mask):
        data_for_interp[nan_mask] = 0

    warped = ndimage.map_coordinates(
        data_for_interp, coords, order=order, mode="constant", cval=0
    )
    warped[~valid] = np.nan

    bounds = (lon_min, lat_min, lon_max, lat_max)
    log.info(f"    Warped to {out_width}x{out_height}, bounds: {bounds}")
    return warped, bounds, valid


# ---------------------------------------------------------------------------
# Step 5: Vectorize RFI masks to GeoJSON polygons
# ---------------------------------------------------------------------------

def vectorize_rfi_mask(
    bright_mask: np.ndarray,
    bounds: tuple,
    valid_mask: np.ndarray,
    product_name: str,
    polarization: str,
    date: str,
    score: float,
    severity: str,
    min_area_deg2: float = 1e-4,
) -> list:
    """
    Convert a warped boolean RFI mask to GeoJSON feature dicts.

    Uses rasterio.features.shapes() with the geographic transform derived from bounds.
    """
    from rasterio.transform import from_bounds
    from rasterio.features import shapes
    from shapely.geometry import shape, mapping
    from shapely.validation import make_valid

    lon_min, lat_min, lon_max, lat_max = bounds
    h, w = bright_mask.shape

    # Apply valid mask and dilate to cluster nearby pixels
    mask = bright_mask & valid_mask
    mask = ndimage.binary_dilation(mask, iterations=3)

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)
    # Note: from_bounds expects (west, south, east, north, width, height)
    # but rasterio features.shapes expects the transform to map from pixel to CRS
    # For a top-to-bottom image, we need the affine that goes from pixel coords
    # to geographic coords where row 0 = lat_max

    mask_uint8 = mask.astype(np.uint8)
    features_list = []

    for geom, value in shapes(mask_uint8, mask=(mask_uint8 == 1), transform=transform):
        if value == 0:
            continue
        poly = shape(geom)
        poly = make_valid(poly)
        if poly.is_empty or poly.area < min_area_deg2:
            continue
        # Simplify geometry
        poly = poly.simplify(0.005, preserve_topology=True)
        if poly.is_empty:
            continue

        features_list.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "product": product_name,
                "polarization": polarization,
                "date": date,
                "rfi_score": round(score, 1),
                "severity": severity,
            },
        })

    log.info(f"    Vectorized {len(features_list)} RFI polygons")
    return features_list


# ---------------------------------------------------------------------------
# Step 6: Encode SAR overlays as base64 PNG
# ---------------------------------------------------------------------------

def sar_to_base64_png(
    warped_db: np.ndarray,
    valid_mask: np.ndarray,
    alpha_inside: int = 180,
) -> str:
    """Convert warped SAR dB data to a base64-encoded RGBA PNG string."""
    h, w = warped_db.shape

    # Percentile stretch on valid pixels
    valid_vals = warped_db[valid_mask & np.isfinite(warped_db)]
    if len(valid_vals) == 0:
        return ""
    p2, p98 = np.percentile(valid_vals, [2, 98])
    stretched = np.clip((warped_db - p2) / max(p98 - p2, 0.01), 0, 1)
    np.nan_to_num(stretched, copy=False, nan=0.0)
    gray = (stretched * 255).astype(np.uint8)

    # Build RGBA
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = gray
    rgba[:, :, 1] = gray
    rgba[:, :, 2] = gray
    rgba[:, :, 3] = np.where(valid_mask, alpha_inside, 0).astype(np.uint8)

    img = Image.fromarray(rgba)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    size_kb = len(buf.getvalue()) / 1024
    log.info(f"    PNG size: {size_kb:.0f} KB")
    return b64


# ---------------------------------------------------------------------------
# Step 7: Assemble single HTML file
# ---------------------------------------------------------------------------

def build_html(layers: list, report_data: list) -> str:
    """
    Build the Leaflet HTML with all overlays embedded.

    layers: list of dicts with keys:
        label, b64_png, bounds, geojson_features, date, pol, score, severity, default_on
    """
    # Build overlay JS
    overlay_js_parts = []
    geojson_js_parts = []
    overlay_names = {}
    default_on_layers = []

    for i, layer in enumerate(layers):
        lbl = layer["label"]
        b = layer["bounds"]  # (lon_min, lat_min, lon_max, lat_max)
        # Leaflet bounds: [[south, west], [north, east]]
        bounds_js = f"[[{b[1]:.6f}, {b[0]:.6f}], [{b[3]:.6f}, {b[2]:.6f}]]"

        # SAR image overlay
        sar_var = f"sar_{i}"
        overlay_js_parts.append(
            f'var {sar_var} = L.imageOverlay('
            f'"data:image/png;base64,{layer["b64_png"]}", {bounds_js}'
            f');'
        )
        overlay_names[f"SAR: {lbl}"] = sar_var

        # GeoJSON RFI polygons
        if layer["geojson_features"]:
            geojson_var = f"rfi_{i}"
            geojson_data = json.dumps({
                "type": "FeatureCollection",
                "features": layer["geojson_features"],
            })
            geojson_js_parts.append(
                f'var {geojson_var} = L.geoJSON({geojson_data}, {{\n'
                f'  style: function(feature) {{\n'
                f'    var s = feature.properties.severity;\n'
                f'    var color = s === "HIGH" ? "#ff0000" : s === "MODERATE" ? "#ff8c00" : "#ffd700";\n'
                f'    return {{color: color, weight: 1.5, fillColor: color, fillOpacity: 0.3, opacity: 0.8}};\n'
                f'  }},\n'
                f'  onEachFeature: function(feature, layer) {{\n'
                f'    var p = feature.properties;\n'
                f'    layer.bindPopup(\n'
                f'      "<b>RFI Detection</b><br>"\n'
                f'      + "Product: " + p.product + "<br>"\n'
                f'      + "Date: " + p.date + "<br>"\n'
                f'      + "Polarization: " + p.polarization + "<br>"\n'
                f'      + "RFI Score: " + p.rfi_score + "/100<br>"\n'
                f'      + "Severity: <b>" + p.severity + "</b>"\n'
                f'    );\n'
                f'  }}\n'
                f'}});'
            )
            overlay_names[f"RFI: {lbl}"] = geojson_var

        if layer.get("default_on"):
            default_on_layers.append(sar_var)
            if layer["geojson_features"]:
                default_on_layers.append(geojson_var)

    # Build layer control object
    overlays_obj_entries = [f'    "{k}": {v}' for k, v in overlay_names.items()]
    overlays_obj = "{\n" + ",\n".join(overlays_obj_entries) + "\n  }"

    # Default layers to add to map
    default_add = "\n  ".join(f"{v}.addTo(map);" for v in default_on_layers)

    # Compute bounding box of all layers
    all_lats = []
    all_lons = []
    for layer in layers:
        b = layer["bounds"]
        all_lats.extend([b[1], b[3]])
        all_lons.extend([b[0], b[2]])
    sw_lat, ne_lat = min(all_lats), max(all_lats)
    sw_lon, ne_lon = min(all_lons), max(all_lons)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Sentinel-1 RFI / GPS Jamming Detection</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body {{ margin: 0; padding: 0; }}
    #map {{ width: 100%; height: 100vh; }}
    .legend {{
      background: white;
      padding: 10px 14px;
      border-radius: 5px;
      box-shadow: 0 0 15px rgba(0,0,0,0.2);
      font: 13px/1.5 Arial, sans-serif;
      max-width: 200px;
    }}
    .legend h4 {{ margin: 0 0 8px 0; font-size: 14px; }}
    .legend-item {{ display: flex; align-items: center; margin-bottom: 4px; }}
    .legend-color {{
      width: 20px; height: 14px;
      margin-right: 8px;
      border: 1px solid #666;
      flex-shrink: 0;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script>
  var map = L.map('map').fitBounds([[{sw_lat:.4f}, {sw_lon:.4f}], [{ne_lat:.4f}, {ne_lon:.4f}]]);

  var cartoVoyager = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
    maxZoom: 19,
    subdomains: 'abcd'
  }}).addTo(map);

  var esriImagery = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
    attribution: '&copy; <a href="https://www.esri.com/">Esri</a>, Maxar, Earthstar Geographics',
    maxZoom: 19
  }});

  var baseMaps = {{
    "Map": cartoVoyager,
    "Satellite": esriImagery
  }};

  // SAR image overlays
  {"  ".join(overlay_js_parts)}

  // RFI polygon layers
  {"  ".join(geojson_js_parts)}

  // Default layers
  {default_add}

  // Layer control
  L.control.layers(baseMaps, {overlays_obj}, {{collapsed: false}}).addTo(map);

  // Legend
  var legend = L.control({{position: 'bottomright'}});
  legend.onAdd = function(map) {{
    var div = L.DomUtil.create('div', 'legend');
    div.innerHTML = '<h4>RFI Severity</h4>'
      + '<div class="legend-item"><div class="legend-color" style="background:#ff0000;opacity:0.5;"></div>HIGH (&gt;60)</div>'
      + '<div class="legend-item"><div class="legend-color" style="background:#ff8c00;opacity:0.5;"></div>MODERATE (30-60)</div>'
      + '<div class="legend-item"><div class="legend-color" style="background:#ffd700;opacity:0.5;"></div>LOW (10-30)</div>'
      + '<div class="legend-item"><div class="legend-color" style="background:#888;opacity:0.3;"></div>SAR overlay</div>';
    return div;
  }};
  legend.addTo(map);
  </script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def find_products() -> list:
    """Discover downloaded .SAFE directories and their files."""
    products = []
    for safe_dir in sorted(DOWNLOADS_DIR.glob("*.SAFE")):
        product_name = safe_dir.stem
        annotation_dir = safe_dir / "annotation"
        measurement_dir = safe_dir / "measurement"
        if not annotation_dir.exists() or not measurement_dir.exists():
            continue

        # Find VV and VH pairs
        for pol in ["vv", "vh"]:
            # Find annotation XML (not rfi/ or calibration/)
            ann_xmls = [
                f for f in annotation_dir.glob("*.xml")
                if f.name.startswith("s1") and f"-{pol}-" in f.name
            ]
            tif_files = [
                f for f in measurement_dir.glob("*.tiff")
                if f"-{pol}-" in f.name
            ]
            if ann_xmls and tif_files:
                products.append({
                    "product_name": product_name,
                    "safe_dir": safe_dir,
                    "polarization": pol.upper(),
                    "annotation_xml": ann_xmls[0],
                    "tif_path": tif_files[0],
                })
    return products


def main():
    log.info("=" * 60)
    log.info("Creating interactive RFI detection map")
    log.info("=" * 60)

    # Load existing RFI report
    report_path = OUTPUT_DIR / "rfi_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report_data = json.load(f)
        log.info(f"Loaded RFI report with {len(report_data)} entries")
    else:
        report_data = []
        log.warning("No rfi_report.json found, scores will be computed fresh")

    # Build lookup from report
    report_lookup = {}
    for entry in report_data:
        key = (entry["product_name"], entry["polarization"])
        report_lookup[key] = entry

    # Discover products
    products = find_products()
    log.info(f"Found {len(products)} product/polarization combinations")
    if not products:
        log.error("No .SAFE products found in output/downloads/")
        sys.exit(1)

    # Cache parsed grids per SAFE dir (VV and VH share the same grid)
    grid_cache = {}
    interp_cache = {}

    layers = []

    for prod in products:
        pname = prod["product_name"]
        pol = prod["polarization"]
        log.info(f"\nProcessing {pname} {pol}")

        # --- Step 1: Parse geolocation grid ---
        safe_key = str(prod["safe_dir"])
        if safe_key not in grid_cache:
            log.info("  Parsing geolocation grid...")
            grid = parse_geolocation_grid(prod["annotation_xml"])
            lat_interp, lon_interp = build_forward_interpolators(grid)
            grid_cache[safe_key] = grid
            interp_cache[safe_key] = (lat_interp, lon_interp)
        else:
            grid = grid_cache[safe_key]
            lat_interp, lon_interp = interp_cache[safe_key]

        # --- Step 2: Load SAR at reduced resolution ---
        log.info("  Loading SAR image (subsampled)...")
        data, full_h, full_w = load_sar_subsampled(prod["tif_path"])

        # --- Step 3: RFI detection ---
        log.info("  Running RFI detection...")
        data_db = intensity_to_db(data)
        bright_mask = detect_rfi_bright_pixels(data_db)

        # Get score from report or compute
        report_key = (pname, pol)
        if report_key in report_lookup:
            score = report_lookup[report_key]["score"]
            severity = report_lookup[report_key]["severity"]
            date = report_lookup[report_key]["date"]
        else:
            score = 0.0
            severity = "UNKNOWN"
            # Extract date from product name
            parts = pname.split("_")
            date = parts[4][:8] if len(parts) > 4 else "unknown"
            date = f"{date[:4]}-{date[4:6]}-{date[6:8]}" if len(date) == 8 else date

        # --- Step 4: Warp to geographic ---
        log.info("  Warping SAR overlay to geographic coordinates...")
        warped_db, bounds, valid_mask = warp_to_geographic(
            data_db, lat_interp, lon_interp, full_h, full_w, grid,
            out_width=SAR_OVERLAY_WIDTH, order=1,
        )

        log.info("  Warping RFI mask to geographic coordinates...")
        # Convert bright_mask to float for interpolation, then threshold
        bright_float = bright_mask.astype(np.float64)
        warped_bright, rfi_bounds, rfi_valid = warp_to_geographic(
            bright_float, lat_interp, lon_interp, full_h, full_w, grid,
            out_width=RFI_MASK_WIDTH, order=0,
        )
        warped_bright_bool = (warped_bright > 0.5) & rfi_valid

        # --- Step 5: Vectorize RFI mask ---
        log.info("  Vectorizing RFI polygons...")
        geojson_features = vectorize_rfi_mask(
            warped_bright_bool, rfi_bounds, rfi_valid,
            pname, pol, date, score, severity,
        )

        # --- Step 6: Encode SAR as base64 PNG ---
        log.info("  Encoding SAR overlay as PNG...")
        b64_png = sar_to_base64_png(warped_db, valid_mask)

        # Determine if this should be on by default (Feb 19 VH = strongest)
        is_default = ("20260219" in pname and pol == "VH")

        # Determine region from geographic location
        center_lat = (grid["lat_min"] + grid["lat_max"]) / 2
        if center_lat > 60:
            region = "Norway"
        elif center_lat < 0:
            region = "Madagascar"
        else:
            region = "Iran"

        label = f"{region} {date} {pol}"
        layers.append({
            "label": label,
            "b64_png": b64_png,
            "bounds": bounds,
            "geojson_features": geojson_features,
            "date": date,
            "pol": pol,
            "score": score,
            "severity": severity,
            "default_on": is_default,
        })

    # --- Step 7: Build HTML ---
    log.info("\nAssembling HTML...")
    html = build_html(layers, report_data)

    out_path = OUTPUT_DIR / "rfi_map.html"
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    log.info(f"\nWrote {out_path} ({size_mb:.1f} MB)")
    log.info("Open in a browser to view the interactive map.")


if __name__ == "__main__":
    main()
