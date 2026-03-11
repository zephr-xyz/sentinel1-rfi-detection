#!/usr/bin/env python3
"""Generate spy-styled Leaflet map for Iran RFI detection results.
Pre-bins points at each zoom level in Python for fast browser rendering."""
import json
import math
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output" / "iran_rfi"

# Iran simplified boundary (lon, lat pairs)
AOI_POLYGON = [
    [44.0, 39.4], [44.0, 37.7], [44.4, 36.0], [45.4, 34.0],
    [46.4, 32.0], [48.0, 30.5], [49.6, 29.4], [51.0, 28.0],
    [54.0, 26.6], [57.8, 25.6], [61.6, 25.2], [63.3, 27.2],
    [63.3, 34.1], [61.2, 35.6], [59.6, 36.7], [57.3, 37.6],
    [53.9, 37.1], [50.3, 37.4], [48.6, 38.4], [44.0, 39.4],
]
AOI_LEAFLET = [[lat, lon] for lon, lat in AOI_POLYGON]

GRID_SIZES = {
    5: 0.5, 6: 0.25, 7: 0.1, 8: 0.05, 9: 0.025,
    10: 0.01, 11: 0.005,
}


def load_data():
    temporal = OUTPUT_DIR / "rfi_temporal.json"
    if temporal.exists():
        data = json.load(open(temporal))
        scenes = data.get("scenes", [])
        if scenes:
            print(f"Using temporal z-score data ({data.get('method', '?')})")
            return scenes

    for name in ["rfi_points.json", "rfi_progress.json"]:
        p = OUTPUT_DIR / name
        if p.exists():
            data = json.load(open(p))
            scenes = data.get("scenes", [])
            if scenes:
                return scenes
    return []


def prebin_points(points, cell_size):
    """Bin points into grid cells and compute display properties."""
    bins = {}
    for lat, lon in points:
        r = int(math.floor(lat / cell_size))
        c = int(math.floor(lon / cell_size))
        key = (r, c)
        if key not in bins:
            bins[key] = 0
        bins[key] += 1

    if not bins:
        return []

    counts = sorted(bins.values())
    p85 = counts[int(len(counts) * 0.85)] if counts else 1
    p95 = counts[int(len(counts) * 0.95)] if counts else 1
    max_count = counts[-1] if counts else 1

    # Compact format: [latMin, lonMin, count, colorIdx, opacity100, score]
    # colorIdx: 0=green, 1=orange, 2=red
    result = []
    for (r, c), cnt in bins.items():
        if cnt >= p95:
            ci = 2
        elif cnt >= p85:
            ci = 1
        else:
            ci = 0
        opacity = max(15, min(80, int(cnt / max(1, p95) * 70)))
        score = min(100, int(math.log(1 + cnt) / math.log(1 + max_count) * 100))
        lat_min = round(r * cell_size, 6)
        lon_min = round(c * cell_size, 6)
        result.append([lat_min, lon_min, cnt, ci, opacity, score])

    return result


def generate_map(scenes):
    dates = sorted(set(s["meta"]["date"] for s in scenes if s.get("meta")))
    if not dates:
        print("No scene data to map.")
        return

    date_scenes = {d: [] for d in dates}
    for s in scenes:
        d = s["meta"]["date"]
        if d in date_scenes:
            date_scenes[d].append(s)

    date_stats = {}
    for d in dates:
        ss = date_scenes[d]
        date_stats[d] = {
            "scenes": len(ss),
            "max_score": max((s["score"] for s in ss), default=0),
            "total_bright": sum(s.get("n_bright", 0) for s in ss),
            "total_points": sum(len(s.get("points", [])) for s in ss),
        }

    # Collect raw points per date
    date_raw_points = {}
    for d in dates:
        pts = []
        for s in date_scenes[d]:
            for pt in s.get("points", []):
                pts.append((pt[0], pt[1]))
        date_raw_points[d] = pts

    total_points = sum(len(v) for v in date_raw_points.values())

    # Pre-bin at each zoom level
    print(f"Pre-binning {total_points:,} points at {len(GRID_SIZES)} zoom levels...")
    # Structure: {date: {zoom: [[latMin, lonMin, count, colorIdx, opacity100, score], ...]}}
    date_bins = {}
    for d in dates:
        date_bins[d] = {}
        pts = date_raw_points[d]
        for zoom, cs in GRID_SIZES.items():
            date_bins[d][zoom] = prebin_points(pts, cs)
        print(f"  {d}: {len(pts):,} pts -> z6:{len(date_bins[d][6])} z8:{len(date_bins[d][8])} z10:{len(date_bins[d][10])} z11:{len(date_bins[d][11])} bins")

    center_lat = sum(p[0] for p in AOI_LEAFLET) / len(AOI_LEAFLET)
    center_lon = sum(p[1] for p in AOI_LEAFLET) / len(AOI_LEAFLET)

    # Serialize bins compactly
    grid_sizes_js = json.dumps(GRID_SIZES)
    date_bins_js = json.dumps(date_bins, separators=(',', ':'))

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SIGINT // IRAN RFI SURVEILLANCE</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #000; overflow: hidden; }}
    #map {{ width: 100vw; height: 100vh; background: #0a0a0a; }}

    #scanlines {{
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,136,0.015) 2px, rgba(0,255,136,0.015) 4px);
        pointer-events: none; z-index: 9999;
    }}

    #hud-header {{
        position: fixed; top: 0; right: 0; z-index: 1000;
        background: linear-gradient(180deg, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0.6) 70%, transparent 100%);
        padding: 12px 20px 30px 20px;
        font-family: 'Share Tech Mono', 'Courier New', monospace;
        border-bottom: 1px solid rgba(0,255,136,0.2);
        text-align: right;
    }}
    #hud-header .title {{
        color: #00ff88; font-size: 14px; letter-spacing: 4px;
        text-shadow: 0 0 10px rgba(0,255,136,0.5);
    }}
    #hud-header .subtitle {{
        color: #667; font-size: 10px; letter-spacing: 2px; margin-top: 2px;
    }}

    #timeline {{
        position: fixed; bottom: 0; left: 0; right: 0; z-index: 1000;
        background: linear-gradient(0deg, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0.7) 70%, transparent 100%);
        padding: 30px 20px 15px 20px;
        font-family: 'Share Tech Mono', 'Courier New', monospace;
    }}
    #timeline .date-display {{
        color: #00ff88; font-size: 20px; text-align: center;
        letter-spacing: 3px; margin-bottom: 8px;
        text-shadow: 0 0 15px rgba(0,255,136,0.6);
    }}
    #timeline .stats {{
        color: #556; font-size: 10px; text-align: center;
        letter-spacing: 2px; margin-bottom: 10px;
    }}
    #timeline input[type=range] {{
        width: 100%; -webkit-appearance: none; appearance: none;
        height: 3px; background: #1a1a1a; outline: none;
        border: 1px solid #333;
    }}
    #timeline input[type=range]::-webkit-slider-thumb {{
        -webkit-appearance: none; appearance: none;
        width: 16px; height: 16px; background: #00ff88;
        border-radius: 50%; cursor: pointer;
        box-shadow: 0 0 10px rgba(0,255,136,0.8), 0 0 20px rgba(0,255,136,0.4);
    }}
    #timeline .controls {{
        display: flex; justify-content: center; gap: 10px; margin-top: 8px;
    }}
    #timeline button {{
        background: transparent; border: 1px solid #00ff88; color: #00ff88;
        font-family: 'Share Tech Mono', monospace; font-size: 11px;
        padding: 4px 16px; cursor: pointer; letter-spacing: 2px;
        transition: all 0.2s;
    }}
    #timeline button:hover {{ background: rgba(0,255,136,0.15); }}
    #timeline button.active {{ background: rgba(0,255,136,0.25); }}

    .leaflet-control-layers {{
        background: rgba(10,10,10,0.95) !important;
        border: 1px solid rgba(0,255,136,0.3) !important;
        color: #00ff88 !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 11px !important;
        border-radius: 0 !important;
        box-shadow: 0 0 20px rgba(0,0,0,0.8) !important;
    }}
    .leaflet-control-layers-expanded {{ padding: 8px 12px !important; }}
    .leaflet-control-layers label {{ color: #aab !important; }}
    .leaflet-control-layers-separator {{ border-color: rgba(0,255,136,0.2) !important; }}

    #info-panel {{
        position: fixed; top: 65px; right: 10px; z-index: 1000;
        background: rgba(10,10,10,0.92); border: 1px solid rgba(0,255,136,0.25);
        padding: 10px 14px; font-family: 'Share Tech Mono', monospace;
        font-size: 10px; color: #667; max-width: 220px;
        letter-spacing: 1px; line-height: 1.6;
    }}
    #info-panel .label {{ color: #445; }}
    #info-panel .value {{ color: #00ff88; }}
    #info-panel .warn {{ color: #ff6600; }}
    #info-panel .crit {{ color: #ff0040; }}
    #opacity-slider::-webkit-slider-thumb, #score-slider::-webkit-slider-thumb {{
        -webkit-appearance: none; appearance: none;
        width: 12px; height: 12px; background: #00ff88;
        border-radius: 50%; cursor: pointer;
        box-shadow: 0 0 6px rgba(0,255,136,0.6);
    }}

    .leaflet-popup-content-wrapper {{
        background: transparent !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }}
    .leaflet-popup-content {{ margin: 0 !important; }}
    .leaflet-popup-tip {{ display: none; }}

    .corner {{ position: fixed; z-index: 999; pointer-events: none; }}
    .corner-tl {{ top: 55px; left: 10px; border-top: 2px solid rgba(0,255,136,0.3); border-left: 2px solid rgba(0,255,136,0.3); width: 30px; height: 30px; }}
    .corner-tr {{ top: 55px; right: 10px; border-top: 2px solid rgba(0,255,136,0.3); border-right: 2px solid rgba(0,255,136,0.3); width: 30px; height: 30px; }}
    .corner-bl {{ bottom: 90px; left: 10px; border-bottom: 2px solid rgba(0,255,136,0.3); border-left: 2px solid rgba(0,255,136,0.3); width: 30px; height: 30px; }}
    .corner-br {{ bottom: 90px; right: 10px; border-bottom: 2px solid rgba(0,255,136,0.3); border-right: 2px solid rgba(0,255,136,0.3); width: 30px; height: 30px; }}
</style>
</head>
<body>
<div id="scanlines"></div>
<div id="hud-header">
    <div class="title">SIGINT // SAR RFI SURVEILLANCE FEED</div>
    <div class="subtitle">SENTINEL-1 C-BAND INTERFERENCE DETECTION // IRAN THEATER</div>
</div>
<div class="corner corner-tl"></div>
<div class="corner corner-tr"></div>
<div class="corner corner-bl"></div>
<div class="corner corner-br"></div>
<div id="info-panel">
    <div style="color:#00ff88;font-size:11px;margin-bottom:6px;border-bottom:1px solid #222;padding-bottom:4px;">// MISSION PARAMS</div>
    <span class="label">SENSOR:</span> <span class="value">SENTINEL-1A/C</span><br>
    <span class="label">BAND:</span> <span class="value">C-BAND 5.405 GHz</span><br>
    <span class="label">MODE:</span> <span class="value">IW GRDH</span><br>
    <span class="label">WINDOW:</span> <span class="value">{dates[0]} - {dates[-1]}</span><br>
    <span class="label">SCENES:</span> <span class="value">{len(scenes)}</span><br>
    <span class="label">RFI POINTS:</span> <span class="value">{total_points:,}</span><br>
    <span class="label">DEM MASK:</span> <span class="value">SLOPE &gt;15deg</span><br>
    <div style="margin-top:6px;border-top:1px solid #222;padding-top:4px;">
    <span class="label">RFI DENSITY:</span><br>
    <span style="display:inline-block;width:10px;height:10px;background:#00ff88;margin:2px 4px 0 0;vertical-align:middle;"></span><span class="label">LOW</span><br>
    <span style="display:inline-block;width:10px;height:10px;background:#ff6600;margin:2px 4px 0 0;vertical-align:middle;"></span><span class="warn">MODERATE (top 15%)</span><br>
    <span style="display:inline-block;width:10px;height:10px;background:#ff0040;margin:2px 4px 0 0;vertical-align:middle;"></span><span class="crit">HIGH (top 5%)</span>
    </div>
    <div style="margin-top:6px;border-top:1px solid #222;padding-top:4px;">
    <span class="label">RFI OPACITY:</span> <span class="value" id="opacity-val">100%</span><br>
    <input type="range" id="opacity-slider" min="0" max="100" value="100" step="5"
        style="width:100%;margin-top:4px;-webkit-appearance:none;appearance:none;height:3px;background:#1a1a1a;outline:none;border:1px solid #333;">
    </div>
    <div style="margin-top:6px;border-top:1px solid #222;padding-top:4px;">
    <span class="label">MIN RFI SCORE:</span> <span class="value" id="score-val">0</span><br>
    <input type="range" id="score-slider" min="0" max="100" value="0" step="1"
        style="width:100%;margin-top:4px;-webkit-appearance:none;appearance:none;height:3px;background:#1a1a1a;outline:none;border:1px solid #333;">
    </div>
</div>
<div id="map"></div>
<div id="timeline">
    <div class="date-display" id="current-date">{dates[0]}</div>
    <div class="stats" id="current-stats">LOADING...</div>
    <input type="range" id="date-slider" min="0" max="{len(dates)-1}" value="0" step="1">
    <div class="controls">
        <button id="btn-prev" onclick="stepDate(-1)">&lt; PREV</button>
        <button id="btn-play" onclick="togglePlay()">PLAY</button>
        <button id="btn-next" onclick="stepDate(1)">NEXT &gt;</button>
        <button id="btn-all" onclick="showAll()">ALL</button>
    </div>
</div>

<script>
    var dates = {json.dumps(dates)};
    var dateStats = {json.dumps(date_stats)};
    var gridSizes = {grid_sizes_js};
    var dateBins = {date_bins_js};

    var COLORS = ['#00ff88', '#ff6600', '#ff0040'];
    var LEVELS = ['LOW', 'MODERATE', 'HIGH'];

    var currentIdx = 0;
    var playing = false;
    var playInterval = null;
    var showAllMode = false;
    var rfiOpacity = 1.0;
    var minScore = 0;

    var canvasRenderer = L.canvas({{ padding: 0.3 }});

    var map = L.map('map', {{
        center: [{center_lat}, {center_lon}],
        zoom: 6,
        zoomControl: false,
        attributionControl: false,
        preferCanvas: true
    }});

    var esriSat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
        maxZoom: 18, opacity: 0.7
    }});
    var cartoDark = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_nolabels/{{z}}/{{x}}/{{y}}{{r}}.png', {{
        maxZoom: 19, opacity: 0.95
    }});
    var cartoLabels = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_only_labels/{{z}}/{{x}}/{{y}}{{r}}.png', {{
        maxZoom: 19, opacity: 0.6
    }});

    esriSat.addTo(map);
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_nolabels/{{z}}/{{x}}/{{y}}{{r}}.png', {{
        maxZoom: 19, opacity: 0.45
    }}).addTo(map);
    cartoLabels.addTo(map);

    var aoiPoly = L.polygon({json.dumps(AOI_LEAFLET)}, {{
        color: '#00ff88', weight: 1.5, fillOpacity: 0, dashArray: '8,4', opacity: 0.5
    }}).addTo(map);

    function getZoomKey(zoom) {{
        if (zoom <= 5) return '5';
        if (zoom >= 11) return '11';
        return '' + zoom;
    }}

    var activeLayers = [];

    function clearActiveLayers() {{
        for (var i = 0; i < activeLayers.length; i++) {{
            map.removeLayer(activeLayers[i]);
        }}
        activeLayers = [];
    }}

    function renderDate(dateStr) {{
        var zk = getZoomKey(map.getZoom());
        var cs = gridSizes[zk];
        var bins = dateBins[dateStr] && dateBins[dateStr][zk];
        if (!bins || bins.length === 0) return;

        // Viewport culling
        var bounds = map.getBounds();
        var south = bounds.getSouth() - cs;
        var north = bounds.getNorth() + cs;
        var west = bounds.getWest() - cs;
        var east = bounds.getEast() + cs;

        var group = L.layerGroup();
        for (var i = 0; i < bins.length; i++) {{
            var b = bins[i];
            // b = [latMin, lonMin, count, colorIdx, opacity100, score]
            var latMin = b[0], lonMin = b[1];
            if (latMin > north || latMin + cs < south || lonMin > east || lonMin + cs < west) continue;
            if (b[5] < minScore) continue;

            var color = COLORS[b[3]];
            var fillOp = (b[4] / 100.0) * rfiOpacity;
            var strokeOp = 0.6 * rfiOpacity;

            var rect = L.rectangle(
                [[latMin, lonMin], [latMin + cs, lonMin + cs]],
                {{ fillColor: color, fillOpacity: fillOp, color: color, weight: 0.5, opacity: strokeOp, renderer: canvasRenderer }}
            );
            var cnt = b[2], scoreVal = b[5], level = LEVELS[b[3]];
            rect.bindPopup(
                '<div style="font-family:Courier New,monospace;color:#00ff88;background:#0a0a0a;padding:8px;border:1px solid #00ff88;font-size:11px;">' +
                '<div style="color:#ff0040;font-weight:bold;margin-bottom:4px;">// SIGNAL INTERCEPT</div>' +
                '<b>DATE:</b> ' + dateStr + '<br>' +
                '<b>RFI SCORE:</b> <span style="color:' + (scoreVal > 75 ? '#ff0040' : scoreVal > 40 ? '#ff6600' : '#00ff88') + '">' + scoreVal + '/100</span><br>' +
                '<b>DENSITY:</b> <span style="color:' + color + '">' + level + '</span><br>' +
                '<b>RFI PIXELS:</b> ' + cnt + '<br>' +
                '<b>GRID:</b> ' + cs.toFixed(4) + '&deg;' +
                '</div>',
                {{className: 'spy-popup'}}
            );
            group.addLayer(rect);
        }}
        group.addTo(map);
        activeLayers.push(group);
    }}

    function renderCurrent() {{
        clearActiveLayers();
        if (showAllMode) {{
            for (var i = 0; i < dates.length; i++) {{
                renderDate(dates[i]);
            }}
        }} else {{
            renderDate(dates[currentIdx]);
        }}
    }}

    map.on('zoomend', function() {{
        renderCurrent();
    }});

    map.on('moveend', function() {{
        renderCurrent();
    }});

    function updateDisplay(idx) {{
        currentIdx = idx;
        var d = dates[idx];
        document.getElementById('current-date').textContent = d;
        document.getElementById('date-slider').value = idx;
        var stats = dateStats[d] || {{scenes: 0, max_score: 0, total_bright: 0, total_points: 0}};
        var threatColor = stats.max_score > 60 ? '#ff0040' : stats.max_score > 30 ? '#ff6600' : '#00ff88';
        document.getElementById('current-stats').innerHTML =
            'PASSES: ' + stats.scenes + ' // PEAK RFI: <span style="color:' + threatColor + '">' +
            stats.max_score.toFixed(0) + '/100</span> // DETECTIONS: ' + stats.total_points;
        showAllMode = false;
        document.getElementById('btn-all').classList.remove('active');
        renderCurrent();
    }}

    function stepDate(delta) {{
        var newIdx = Math.max(0, Math.min(dates.length - 1, currentIdx + delta));
        updateDisplay(newIdx);
    }}

    function togglePlay() {{
        playing = !playing;
        var btn = document.getElementById('btn-play');
        if (playing) {{
            btn.textContent = 'PAUSE';
            btn.classList.add('active');
            playInterval = setInterval(function() {{
                var next = (currentIdx + 1) % dates.length;
                updateDisplay(next);
            }}, 2000);
        }} else {{
            btn.textContent = 'PLAY';
            btn.classList.remove('active');
            clearInterval(playInterval);
        }}
    }}

    function showAll() {{
        if (playing) togglePlay();
        showAllMode = !showAllMode;
        var btn = document.getElementById('btn-all');
        if (showAllMode) {{
            btn.classList.add('active');
            renderCurrent();
            document.getElementById('current-date').textContent = 'ALL DATES';
            document.getElementById('current-stats').textContent = 'SHOWING ALL DATES';
        }} else {{
            btn.classList.remove('active');
            updateDisplay(currentIdx);
        }}
    }}

    document.getElementById('date-slider').addEventListener('input', function(e) {{
        updateDisplay(parseInt(e.target.value));
    }});

    document.getElementById('opacity-slider').addEventListener('input', function(e) {{
        rfiOpacity = parseInt(e.target.value) / 100.0;
        document.getElementById('opacity-val').textContent = e.target.value + '%';
        renderCurrent();
    }});

    document.getElementById('score-slider').addEventListener('input', function(e) {{
        minScore = parseInt(e.target.value);
        document.getElementById('score-val').textContent = e.target.value;
        renderCurrent();
    }});

    var baseMaps = {{"SATELLITE": esriSat, "DARK": cartoDark}};
    var overlayMaps = {{"AOI BOUNDARY": aoiPoly, "LABELS": cartoLabels}};
    L.control.layers(baseMaps, overlayMaps, {{collapsed: false, position: 'topleft'}}).addTo(map);
    L.control.zoom({{position: 'topleft'}}).addTo(map);

    updateDisplay(0);
</script>
</body>
</html>"""

    map_path = OUTPUT_DIR / "iran_rfi_map.html"
    with open(map_path, "w") as f:
        f.write(html)
    print(f"Map saved to {map_path}")
    print(f"  Dates: {dates}")
    print(f"  Scenes: {len(scenes)}")
    print(f"  Total points: {total_points:,}")


def main():
    scenes = load_data()
    if not scenes:
        print("No RFI data found yet. Run iran_download_process.py first.")
        return
    generate_map(scenes)


if __name__ == "__main__":
    main()
