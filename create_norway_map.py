#!/usr/bin/env python3
"""Generate spy-styled Leaflet map for Norway Jammertest RFI detection results."""
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output" / "jamertest"

BLEIK_LAT, BLEIK_LON = 69.27, 15.86

# Jammertest 2025 transmitter sites (from NPRA/jammertest-plan)
JAMMER_SITES = [
    {"name": "Bleik", "lat": 69.2726, "lon": 15.9554, "desc": "Meaconing/spoofing antennas"},
    {"name": "Ramnan", "lat": 69.2480, "lon": 15.9200, "desc": "Porcus Maior 50W PRN jammer"},
    {"name": "Stave", "lat": 69.2630, "lon": 15.8100, "desc": "Site 3 test/meeting area"},
]

# Norway Jammertest AOI (rough bounding box around scenes)
AOI_POLYGON = [
    [10.0, 67.0], [22.0, 67.0], [22.0, 72.0], [10.0, 72.0], [10.0, 67.0],
]
AOI_LEAFLET = [[lat, lon] for lon, lat in AOI_POLYGON]


def load_data():
    temporal = OUTPUT_DIR / "rfi_temporal.json"
    if temporal.exists():
        data = json.load(open(temporal))
        scenes = data.get("scenes", [])
        if scenes:
            print(f"Using temporal z-score data ({data.get('method', '?')})")
            return scenes
    return []


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

    date_points = {}
    for d in dates:
        all_pts = []
        for s in date_scenes[d]:
            for pt in s.get("points", []):
                all_pts.append([pt[0], pt[1]])
        date_points[d] = all_pts

    total_points = sum(len(v) for v in date_points.values())

    # Build scene info for the info panel
    scene_notes = {}
    for s in scenes:
        d = s["meta"]["date"]
        if d not in scene_notes:
            scene_notes[d] = s.get("note", "")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SIGINT // NORWAY JAMMERTEST RFI</title>
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
        font-size: 10px; color: #667; max-width: 240px;
        letter-spacing: 1px; line-height: 1.6;
    }}
    #info-panel .label {{ color: #445; }}
    #info-panel .value {{ color: #00ff88; }}
    #info-panel .warn {{ color: #ff6600; }}
    #info-panel .crit {{ color: #ff0040; }}

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
    <div class="subtitle">SENTINEL-1 C-BAND INTERFERENCE DETECTION // NORWAY JAMMERTEST 2025</div>
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
    <span class="label">METHOD:</span> <span class="value">TEMPORAL Z-SCORE</span><br>
    <span class="label">WINDOW:</span> <span class="value">{dates[0]} - {dates[-1]}</span><br>
    <span class="label">SCENES:</span> <span class="value">{len(scenes)}</span><br>
    <span class="label">RFI POINTS:</span> <span class="value">{total_points:,}</span><br>
    <span class="label">DEM MASK:</span> <span class="value">SLOPE &gt;15deg</span><br>
    <div style="margin-top:6px;border-top:1px solid #222;padding-top:4px;">
    <span class="label">JAMMER SITES:</span><br>
    <span class="crit">&bull; BLEIK</span> <span class="label">69.27N 15.96E</span><br>
    <span class="crit">&bull; RAMNAN</span> <span class="label">69.25N 15.92E</span><br>
    <span class="crit">&bull; STAVE</span> <span class="label">69.26N 15.81E</span><br>
    <span class="label">EVENT:</span> <span class="value">JAMMERTEST 2025</span><br>
    <span class="label">DATES:</span> <span class="value">Sep 15-19</span><br>
    </div>
    <div style="margin-top:6px;border-top:1px solid #222;padding-top:4px;">
    <span class="label">RFI DENSITY:</span><br>
    <span style="display:inline-block;width:10px;height:10px;background:#00ff88;margin:2px 4px 0 0;vertical-align:middle;"></span><span class="label">LOW</span><br>
    <span style="display:inline-block;width:10px;height:10px;background:#ff6600;margin:2px 4px 0 0;vertical-align:middle;"></span><span class="warn">MODERATE (top 15%)</span><br>
    <span style="display:inline-block;width:10px;height:10px;background:#ff0040;margin:2px 4px 0 0;vertical-align:middle;"></span><span class="crit">HIGH (top 5%)</span>
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
    var datePoints = {json.dumps(date_points)};
    var sceneNotes = {json.dumps(scene_notes)};

    var currentIdx = 0;
    var playing = false;
    var playInterval = null;
    var showAllMode = false;

    var map = L.map('map', {{
        center: [{BLEIK_LAT}, {BLEIK_LON}],
        zoom: 8,
        zoomControl: false,
        attributionControl: false
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

    // Jammer site markers
    var jammerSites = {json.dumps(JAMMER_SITES)};
    var jammerMarkers = L.layerGroup();
    for (var j = 0; j < jammerSites.length; j++) {{
        var site = jammerSites[j];
        var m = L.circleMarker([site.lat, site.lon], {{
            radius: 8, color: '#ffffff', fillColor: '#ffffff', fillOpacity: 0.9, weight: 2
        }});
        m.bindPopup(
            '<div style="font-family:Courier New,monospace;color:#ffffff;background:#0a0a0a;padding:8px;border:1px solid #ffffff;font-size:11px;">' +
            '<div style="font-weight:bold;margin-bottom:4px;">// JAMMER SITE</div>' +
            '<b>LOCATION:</b> ' + site.name.toUpperCase() + ' / AND&Oslash;YA<br>' +
            '<b>COORDS:</b> ' + site.lat.toFixed(4) + 'N, ' + site.lon.toFixed(4) + 'E<br>' +
            '<b>EQUIPMENT:</b> ' + site.desc + '<br>' +
            '<b>EVENT:</b> JAMMERTEST 2025<br>' +
            '<b>DATES:</b> Sep 15-19, 2025<br>' +
            '<b>HOURS:</b> 09:00-15:00 local' +
            '</div>',
            {{className: 'spy-popup'}}
        );
        // Label
        var label = L.marker([site.lat, site.lon], {{
            icon: L.divIcon({{
                className: '',
                html: '<div style="font-family:Share Tech Mono,monospace;color:#ffffff;font-size:10px;text-shadow:0 0 4px #000,0 0 8px #000;white-space:nowrap;pointer-events:none;margin-left:12px;margin-top:-6px;">' + site.name.toUpperCase() + '</div>',
                iconSize: [0, 0]
            }})
        }});
        jammerMarkers.addLayer(m);
        jammerMarkers.addLayer(label);
    }}
    jammerMarkers.addTo(map);

    // Range rings centered on cluster centroid
    var clusterLat = (69.2726 + 69.2480 + 69.2630) / 3;
    var clusterLon = (15.9554 + 15.9200 + 15.8100) / 3;
    var bleikRing20 = L.circle([clusterLat, clusterLon], {{
        radius: 20000, color: '#ffffff', weight: 1.5, fillOpacity: 0, dashArray: '6,4', opacity: 0.7
    }}).addTo(map);
    var bleikRing50 = L.circle([clusterLat, clusterLon], {{
        radius: 50000, color: '#ffffff', weight: 1, fillOpacity: 0, dashArray: '6,4', opacity: 0.4
    }}).addTo(map);

    function gridSize(zoom) {{
        if (zoom <= 6) return 0.25;
        if (zoom <= 7) return 0.1;
        if (zoom <= 8) return 0.05;
        if (zoom <= 9) return 0.025;
        if (zoom <= 10) return 0.01;
        if (zoom <= 11) return 0.005;
        if (zoom <= 12) return 0.002;
        return 0.001;
    }}

    function binPoints(points, cellSize) {{
        var bins = {{}};
        for (var i = 0; i < points.length; i++) {{
            var lat = points[i][0];
            var lon = points[i][1];
            var r = Math.floor(lat / cellSize);
            var c = Math.floor(lon / cellSize);
            var key = r + ',' + c;
            if (!bins[key]) {{
                bins[key] = {{count: 0, latMin: r * cellSize, lonMin: c * cellSize}};
            }}
            bins[key].count++;
        }}

        var counts = [];
        for (var k in bins) {{ counts.push(bins[k].count); }}
        counts.sort(function(a,b) {{ return a - b; }});
        var p85 = counts[Math.floor(counts.length * 0.85)] || 1;
        var p95 = counts[Math.floor(counts.length * 0.95)] || 1;

        var features = [];
        for (var k in bins) {{
            var b = bins[k];
            var c = b.count;
            var color, level;
            if (c >= p95) {{
                color = '#ff0040'; level = 'HIGH';
            }} else if (c >= p85) {{
                color = '#ff6600'; level = 'MODERATE';
            }} else {{
                color = '#00ff88'; level = 'LOW';
            }}
            var opacity = Math.max(0.15, Math.min(0.8, c / Math.max(1, p95) * 0.7));
            var maxCount = counts[counts.length - 1] || 1;
            var score = Math.round(Math.min(100, Math.log(1 + c) / Math.log(1 + maxCount) * 100));
            features.push({{
                type: 'Feature',
                geometry: {{
                    type: 'Polygon',
                    coordinates: [[
                        [b.lonMin, b.latMin],
                        [b.lonMin + cellSize, b.latMin],
                        [b.lonMin + cellSize, b.latMin + cellSize],
                        [b.lonMin, b.latMin + cellSize],
                        [b.lonMin, b.latMin]
                    ]]
                }},
                properties: {{
                    count: b.count,
                    opacity: opacity,
                    color: color,
                    level: level,
                    score: score
                }}
            }});
        }}
        return features;
    }}

    var activeLayers = [];

    function clearActiveLayers() {{
        for (var i = 0; i < activeLayers.length; i++) {{
            map.removeLayer(activeLayers[i]);
        }}
        activeLayers = [];
    }}

    function renderDate(dateStr) {{
        var pts = datePoints[dateStr] || [];
        var zoom = map.getZoom();
        var cs = gridSize(zoom);
        var features = binPoints(pts, cs);

        var layer = L.geoJSON({{type: 'FeatureCollection', features: features}}, {{
            style: function(f) {{
                var p = f.properties;
                return {{
                    fillColor: p.color,
                    fillOpacity: p.opacity,
                    color: p.color,
                    weight: 0.5,
                    opacity: 0.6
                }};
            }},
            onEachFeature: function(f, layer) {{
                var p = f.properties;
                var scoreColor = p.score > 75 ? '#ff0040' : p.score > 40 ? '#ff6600' : '#00ff88';
                layer.bindPopup(
                    '<div style="font-family:Courier New,monospace;color:#00ff88;background:#0a0a0a;padding:8px;border:1px solid #00ff88;font-size:11px;">' +
                    '<div style="color:#ff0040;font-weight:bold;margin-bottom:4px;">// SIGNAL INTERCEPT</div>' +
                    '<b>DATE:</b> ' + dateStr + '<br>' +
                    '<b>RFI SCORE:</b> <span style="color:' + scoreColor + '">' + p.score + '/100</span><br>' +
                    '<b>DENSITY:</b> <span style="color:' + p.color + '">' + p.level + '</span><br>' +
                    '<b>RFI PIXELS:</b> ' + p.count + '<br>' +
                    '<b>GRID:</b> ' + cs.toFixed(4) + '&deg;' +
                    '</div>',
                    {{className: 'spy-popup'}}
                );
            }}
        }});
        layer.addTo(map);
        activeLayers.push(layer);
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
        // Bring jammer sites and range rings to front (above RFI grid)
        bleikRing50.bringToFront();
        bleikRing20.bringToFront();
        jammerMarkers.eachLayer(function(l) {{ if (l.bringToFront) l.bringToFront(); }});
    }}

    map.on('zoomend', function() {{
        renderCurrent();
    }});

    function updateDisplay(idx) {{
        currentIdx = idx;
        var d = dates[idx];
        document.getElementById('current-date').textContent = d;
        document.getElementById('date-slider').value = idx;
        var stats = dateStats[d] || {{scenes: 0, max_score: 0, total_bright: 0, total_points: 0}};
        var note = sceneNotes[d] || '';
        var threatColor = stats.max_score > 60 ? '#ff0040' : stats.max_score > 30 ? '#ff6600' : '#00ff88';
        document.getElementById('current-stats').innerHTML =
            'PASSES: ' + stats.scenes + ' // PEAK RFI: <span style="color:' + threatColor + '">' +
            stats.max_score.toFixed(1) + '/100</span> // DETECTIONS: ' + stats.total_points +
            (note ? ' // <span style="color:#ff6600">' + note + '</span>' : '');
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
            }}, 3000);
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
            var total = 0;
            for (var i = 0; i < dates.length; i++) total += (datePoints[dates[i]] || []).length;
            document.getElementById('current-stats').textContent = 'SHOWING ALL ' + total + ' DETECTIONS';
        }} else {{
            btn.classList.remove('active');
            updateDisplay(currentIdx);
        }}
    }}

    document.getElementById('date-slider').addEventListener('input', function(e) {{
        updateDisplay(parseInt(e.target.value));
    }});

    var baseMaps = {{"SATELLITE": esriSat, "DARK": cartoDark}};
    var overlayMaps = {{"LABELS": cartoLabels, "JAMMER SITES": jammerMarkers, "20km RING": bleikRing20, "50km RING": bleikRing50}};
    L.control.layers(baseMaps, overlayMaps, {{collapsed: false, position: 'topleft'}}).addTo(map);
    L.control.zoom({{position: 'bottomright'}}).addTo(map);

    updateDisplay(0);
</script>
</body>
</html>"""

    map_path = OUTPUT_DIR / "norway_rfi_map.html"
    with open(map_path, "w") as f:
        f.write(html)
    print(f"Map saved to {map_path}")
    print(f"  Dates: {dates}")
    print(f"  Scenes: {len(scenes)}")
    print(f"  Total points: {total_points:,}")


def main():
    scenes = load_data()
    if not scenes:
        print("No RFI data found. Run temporal_rfi_norway.py first.")
        return
    generate_map(scenes)


if __name__ == "__main__":
    main()
