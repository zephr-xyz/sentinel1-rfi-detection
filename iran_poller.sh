#!/bin/bash
# Iran S1 RFI Poller — queries CDSE for new scenes, downloads, runs pipeline, deploys map.
# Usage: ./iran_poller.sh [interval_minutes]
#   default interval: 60 minutes
# Log: /tmp/iran_poller.log

INTERVAL=${1:-60}
BASEDIR="$(cd "$(dirname "$0")" && pwd)"
DOWNLOAD_DIR="$BASEDIR/output/iran_downloads"
REPO_DIR="/tmp/sentinel1-rfi-repo"
LOGFILE="/tmp/iran_poller.log"
LAST_COUNT_FILE="/tmp/iran_scene_count.txt"
LOCKFILE="/tmp/iran_poller.lock"

cd "$BASEDIR"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') — $1" | tee -a "$LOGFILE"
}

# Lock file to prevent concurrent instances
if [ -f "$LOCKFILE" ]; then
    OLD_PID=$(cat "$LOCKFILE" 2>/dev/null)
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') — Another instance running (PID $OLD_PID), exiting" >> "$LOGFILE"
        exit 0
    else
        rm -f "$LOCKFILE"
    fi
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

# Initialize count from current valid scenes
LAST_COUNT=$(find "$DOWNLOAD_DIR" -name "*.SAFE" -type d 2>/dev/null | while read d; do
    if ls "$d/measurement/"*.tiff >/dev/null 2>&1; then echo x; fi
done | wc -l | tr -d ' ')
echo "$LAST_COUNT" > "$LAST_COUNT_FILE"

log "Iran poller started (checking every ${INTERVAL}m, $LAST_COUNT existing scenes)"

while true; do
    log "=== Poll cycle start ==="

    # Step 1: Query CDSE for new scenes and download them (time-limited)
    log "Querying CDSE and downloading new scenes..."
    python3 -u "$BASEDIR/iran_poller_download.py" >> "$LOGFILE" 2>&1
    DL_EXIT=$?
    if [ $DL_EXIT -ne 0 ]; then
        log "WARNING: Download script exited with code $DL_EXIT"
    fi

    # Step 2: Validate — remove any corrupt .SAFE dirs
    log "Validating downloaded scenes..."
    python3 -u -c "
import rasterio
from pathlib import Path
dl = Path('$DOWNLOAD_DIR')
removed = 0
for safe in sorted(dl.glob('*.SAFE')):
    tiffs = list(safe.glob('measurement/*.tiff'))
    if not tiffs:
        continue
    for t in tiffs:
        try:
            with rasterio.open(t) as src:
                _ = src.read(1, window=rasterio.windows.Window(0,0,10,10))
        except Exception:
            import shutil
            shutil.rmtree(safe)
            print(f'Removed corrupt: {safe.name}')
            removed += 1
            break
print(f'Validation done: {removed} corrupt dirs removed')
" >> "$LOGFILE" 2>&1

    # Step 3: Count valid scenes
    CURRENT_COUNT=$(find "$DOWNLOAD_DIR" -name "*.SAFE" -type d 2>/dev/null | while read d; do
        if ls "$d/measurement/"*.tiff >/dev/null 2>&1; then echo x; fi
    done | wc -l | tr -d ' ')

    log "Valid scenes: $CURRENT_COUNT (was $LAST_COUNT)"

    if [ "$CURRENT_COUNT" -gt "$LAST_COUNT" ] || [ ! -f "$BASEDIR/output/iran_rfi/iran_rfi_map.html" ]; then
        NEW=$((CURRENT_COUNT - LAST_COUNT))
        log "$NEW new scenes. Running temporal analysis..."

        # Step 4: Temporal z-score analysis
        python3 -u "$BASEDIR/temporal_rfi.py" >> "$LOGFILE" 2>&1
        if [ $? -ne 0 ]; then
            log "ERROR: temporal_rfi.py failed"
            tail -5 "$LOGFILE"
        else
            log "Temporal analysis complete"

            # Step 5: Regenerate map
            python3 -u "$BASEDIR/create_iran_map.py" >> "$LOGFILE" 2>&1
            if [ $? -ne 0 ]; then
                log "ERROR: create_iran_map.py failed"
            else
                log "Map regenerated"

                # Step 6: Deploy to GitHub Pages
                cd "$REPO_DIR"
                git checkout gh-pages 2>/dev/null
                cp "$BASEDIR/output/iran_rfi/iran_rfi_map.html" "$REPO_DIR/index.html"
                git add index.html
                git commit -m "Auto-update Iran RFI map — $CURRENT_COUNT scenes ($(date +%Y-%m-%d\ %H:%M))" 2>/dev/null
                git push origin gh-pages 2>/dev/null
                if [ $? -eq 0 ]; then
                    log "Deployed to GitHub Pages ($CURRENT_COUNT scenes)"
                else
                    log "ERROR: git push failed"
                fi
                cd "$BASEDIR"
            fi
        fi

        echo "$CURRENT_COUNT" > "$LAST_COUNT_FILE"
        LAST_COUNT=$CURRENT_COUNT
    else
        log "No new scenes, skipping pipeline"
    fi

    log "Next check in ${INTERVAL}m"
    sleep $((INTERVAL * 60))
done
