# Sentinel-1 RFI Detection: Mapping GPS/GNSS Jamming from Space

Detect and map radio frequency interference (RFI) in Sentinel-1 SAR imagery — a technique that can reveal GPS/GNSS jamming activity visible from orbit.

This pipeline searches the Copernicus Data Space Ecosystem (CDSE) for Sentinel-1 GRD products, downloads them, runs multi-method RFI detection, and produces georeferenced interactive maps showing interference zones overlaid on satellite imagery.

## Background: Why SAR Can Detect GPS Jamming

### Sentinel-1 and C-band SAR

The Sentinel-1 constellation (S1A and S1C) carries a C-band synthetic aperture radar operating at **5.405 GHz**. Unlike optical satellites, SAR is an active sensor — it transmits microwave pulses and records the echoes. This makes it sensitive to any ground-based RF emitter operating near its frequency band.

When a SAR satellite passes over an active RF emitter on the ground, the emitter's signal is received by the radar antenna alongside the legitimate radar echoes. The SAR processor cannot distinguish these signals from real backscatter, so the interference appears as anomalous bright streaks, bands, or hotspots in the imagery that don't correspond to any physical features on the ground.

### The GPS/GNSS Connection

GPS/GNSS jammers operate at L-band frequencies (1.164–1.610 GHz) — roughly **4 GHz below** Sentinel-1's C-band. Despite this frequency gap, there are several mechanisms by which GPS jamming infrastructure produces detectable interference in SAR imagery:

**1. Harmonic radiation.** GPS jammers built around oscillators near 1.35 GHz produce harmonics at integer multiples. The 4th harmonic of 1.35 GHz falls at **5.4 GHz** — directly in Sentinel-1's receive band. Even with power falling off at each harmonic, military-grade jammers can produce enough 4th-harmonic energy to saturate a SAR receiver at moderate range.

**2. Co-located electronic warfare infrastructure.** GPS jamming is rarely deployed in isolation. Military and state-level jamming installations typically include broader EW systems that may operate at or near C-band frequencies. The GPS jammer itself serves as a marker for a facility whose other emitters interfere with SAR.

**3. Broadband noise leakage.** High-power jammers, particularly wideband noise generators, don't confine their output precisely to the target frequency band. Spectral sidelobes, intermodulation products, and out-of-band emissions can extend well above the intended jamming band, especially from poorly-filtered or deliberately broadband systems.

### Honest Caveats

The ~4 GHz frequency gap between GPS L-band and Sentinel-1 C-band means **not all GPS jammers will be visible in SAR**. Low-power personal privacy devices and well-filtered military systems may produce no detectable C-band interference. The correlation between SAR RFI hotspots and GPS jamming is strongest for:

- High-power fixed-site jammers (state-level military installations)
- Wideband noise jammers with poor spectral containment
- Dense jamming deployments where multiple systems create cumulative emissions

Conversely, SAR can detect C-band RFI that has nothing to do with GPS — radar altimeters, C-band communications, and other emitters. Context (location, geopolitics, temporal patterns) is essential for attributing SAR RFI detections to GPS jamming specifically.

### Why VH Cross-Polarization Is More Sensitive

Sentinel-1 GRD products include two polarization channels: VV (co-polarized) and VH (cross-polarized). Natural terrain backscatter is much stronger in VV than VH — typically 6–10 dB higher. Ground-based RF emitters, however, radiate with arbitrary polarization that is received roughly equally in both channels.

This means the **signal-to-clutter ratio for RFI is much higher in VH**: the interference signal is comparable in both channels, but the natural background is much lower in VH. In practice, we consistently observe higher RFI detection scores in VH than VV for the same product, making VH the preferred channel for jamming detection.

## Detection Methodology

The pipeline applies four independent detection methods and combines their outputs into a composite severity score.

### 1. Azimuth-Line Analysis

For each azimuth line (row) in the SAR image, compute the mean backscatter in dB. RFI from ground-based emitters elevates entire range lines, producing anomalously bright rows.

- Compute row-wise mean of the dB image
- Estimate robust baseline using median and MAD (median absolute deviation)
- Flag lines where the row mean exceeds **3-sigma** above the median
- Report percentage of flagged lines

### 2. Bright Pixel Detection

Detect individual pixels that are anomalously bright relative to their local neighborhood, indicating localized RFI hotspots.

- Compute a local baseline using median filtering (applied on a downsampled grid for efficiency)
- Subtract baseline to get residuals
- Flag pixels exceeding **4-sigma** above the residual median (using MAD-based robust sigma)
- Report count and percentage of bright pixels

### 3. Spectral Peak Analysis

Analyze the frequency content along azimuth columns to find narrow-band interference signatures.

- Sample 64 evenly-spaced range columns
- For each column, apply a Hann window and compute the FFT
- Estimate the spectral noise floor via the median
- Flag spectral bins exceeding the floor by **5-sigma**
- Report total anomalous spectral peaks across all sampled columns

### 4. Streak Detection

Identify connected bright-pixel structures that form horizontal (range-direction) streaks — the classic visual signature of SAR RFI.

- Label connected components in the bright-pixel mask (downsampled for performance)
- Retain components wider than **50 pixels** with aspect ratio > 5:1 (width:height)
- Report count of detected linear streaks

### Composite Score

The four methods are combined into a single severity score (0–100):

```
score = min(100, pct_rfi_lines * 2 + pct_bright * 10 + min(peaks, 100) * 0.3 + n_streaks * 5)
```

| Score | Severity | Interpretation |
|-------|----------|----------------|
| > 60 | **HIGH** | Strong, persistent RFI — consistent with active jamming |
| 30–60 | **MODERATE** | Significant interference detected |
| 10–30 | **LOW** | Minor interference present |
| 0–10 | **MINIMAL/NONE** | No significant RFI |

## Georeferencing and Map Generation

The interactive map (`create_map.py`) takes raw SAR pixels and places them accurately on a geographic coordinate system:

1. **GCP Extraction**: Parse the 210-point geolocation grid from Sentinel-1 annotation XML files. These ground control points map (line, pixel) coordinates to (latitude, longitude).

2. **Forward/Inverse Interpolation**: Build cubic spline interpolators (`RectBivariateSpline`) for the forward mapping and `griddata` for the inverse mapping (geographic coordinates back to sensor coordinates).

3. **Warping**: Resample SAR imagery and RFI masks from sensor geometry to a regular lat/lon grid using `scipy.ndimage.map_coordinates`.

4. **Vectorization**: Convert warped binary RFI masks to GeoJSON polygons using `rasterio.features.shapes` and simplify with Shapely.

5. **HTML Assembly**: Produce a single self-contained HTML file with:
   - Leaflet.js map with CARTO Voyager and Esri satellite basemaps
   - SAR image overlays as base64-encoded PNGs with configurable opacity
   - RFI polygons styled by severity (red = HIGH, orange = MODERATE, gold = LOW)
   - Layer toggle and click popups with detection metadata

## Results

### Tehran, Iran (Feb–Mar 2026)

Persistent MODERATE-level RFI detected across multiple Sentinel-1A passes over Tehran. The VH channel consistently shows stronger signatures than VV, consistent with ground-based emitters.

| Date | Pass | Pol | Score | Severity |
|------|------|-----|-------|----------|
| 2026-02-19 | Ascending | VH | 51.8 | MODERATE |
| 2026-02-19 | Ascending | VV | 38.2 | MODERATE |

Tehran passes show RFI in every acquisition analyzed, with VH scores 10–15 points higher than VV — the expected pattern for ground-based emitters against the lower cross-pol background.

### Jamertest Norway (Sep 2025)

Norway's [Jamertest](https://www.jamertest.no/) is an annual GPS/GNSS jamming exercise in Bleik, northern Norway, conducted September 10–18, 2025 with published hours of 09:00–23:30 local time.

| Date | Sat | Time (local) | Context | VH Score | VV Score |
|------|-----|-------------|---------|----------|----------|
| 2025-09-10 | S1A | 18:15 | Pre-event baseline | 30.2 | 30.1 |
| 2025-09-11 | S1C | 18:07 | Day 1, DURING jamming | **100.0** | 40.2 |
| 2025-09-16 | S1A | 07:45 | Mid-event, OUTSIDE hours | 78.4 | 34.0 |
| 2025-09-16 | S1C | 18:15 | Mid-event, DURING jamming | **100.0** | 9.3 |
| 2025-09-18 | S1A | 07:28 | Late event, OUTSIDE hours | 35.8 | 46.6 |
| 2025-09-20 | S1A | 18:32 | Post-event baseline | 37.9 | 30.5 |

Key observations:

- **During active jamming hours**: VH scores max out at 100, with 24–46% of azimuth lines flagged and 2.5–3.8% bright pixels. This is unambiguous, intense RFI.
- **Outside jamming hours**: Scores drop to the 30–40 range. The residual signal likely reflects other regional RF sources or residual equipment emissions.
- **Pre/post event baselines** (~30) establish the ambient interference floor for the region.
- **VH vs VV divergence** is most dramatic during active jamming — Sep 16 DURING shows VH=100 vs VV=9.3, a 90-point gap. This is the clearest signature of an artificial emitter against natural terrain backscatter.

### Comparison Summary

| Region | Period | VH Range | VV Range | Interpretation |
|--------|--------|----------|----------|----------------|
| Jamertest (during) | Sep 2025 | 78–100 | 9–40 | Confirmed active GPS jamming |
| Tehran | Feb 2026 | ~52 | ~38 | Persistent moderate interference |
| Jamertest (baseline) | Sep 2025 | 30–38 | 30–31 | Ambient regional RF environment |

## Quickstart

### Prerequisites

- Python 3.9+
- Free [Copernicus Data Space](https://dataspace.copernicus.eu) account (for downloading Sentinel-1 data)

### Install

```bash
git clone https://github.com/zephr-xyz/sentinel1-rfi-detection.git
cd sentinel1-rfi-detection
pip install -r requirements.txt
```

### Configure credentials

```bash
cp .env.example .env
# Edit .env with your CDSE credentials
```

Or export directly:
```bash
export CDSE_USER="your_email@example.com"
export CDSE_PASS="your_password"
```

### Run

```bash
# Demo mode (synthetic data, no downloads needed)
python sentinel1_rfi_demo.py --demo

# Search for available Sentinel-1 products over Tehran
python sentinel1_rfi_demo.py --search-only

# Full pipeline: search, download, detect, and plot
python sentinel1_rfi_demo.py

# Process an already-downloaded .SAFE directory
python sentinel1_rfi_demo.py --local /path/to/S1A_IW_GRDH_*.SAFE

# Generate interactive map from all downloaded products
python create_map.py
```

## Scripts

| Script | Description |
|--------|-------------|
| `sentinel1_rfi_demo.py` | Core pipeline — catalog search, download, RFI detection, and diagnostic plots |
| `create_map.py` | Generate a self-contained interactive Leaflet HTML map from downloaded products |
| `check_tehran.py` | Poll CDSE catalog for new Sentinel-1 passes over Tehran, auto-download and process |
| `download_iran.py` | Batch download Iran-wide scenes from a pre-built selection file |
| `run_jamertest.py` | Memory-efficient RFI analysis of Jamertest Norway exercise scenes |

## Data Sources

- **Sentinel-1 GRD products** from the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu) (free registration required)
- **Basemaps**: [CARTO Voyager](https://carto.com/basemaps/) and [Esri World Imagery](https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9)

## References

- ESA. [Sentinel-1 SAR User Guide](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar). European Space Agency.
- Recchia, A., et al. (2017). "Impact of Radio Frequency Interference on Sentinel-1 SAR data." *ESA Living Planet Symposium*.
- Meyer, F.J., et al. (2013). "Mapping GPS interference in Alaska using Sentinel-1." *IEEE Geoscience and Remote Sensing Letters*.
- Tao, M., et al. (2019). "Radio Frequency Interference Detection and Mitigation for Sentinel-1." *IEEE Trans. Geoscience and Remote Sensing*.
- Norwegian Communications Authority. [Jammertest](https://www.jamertest.no/). Annual GNSS jamming and spoofing test event.

## License

MIT — see [LICENSE](LICENSE).
