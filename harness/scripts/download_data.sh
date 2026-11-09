#!/usr/bin/env bash
# Download competition files one by one into data/competition/<dir>/ (resumable; skips files already present).
# Needs ~/.kaggle/access_token and the project venv. Run from the repo root:
#   bash harness/scripts/download_data.sh files.txt [parallel]
# files.txt: one competition file path per line (e.g. snapshots/fastapi_11194.tgz).
set -u
LIST="$1"; PAR="${2:-2}"
KAGGLE=".venv/Scripts/kaggle.exe"; [ -f "$KAGGLE" ] || KAGGLE=".venv/bin/kaggle"

dl_one() {
  f="$1"; d="data/competition/$(dirname "$f")"; mkdir -p "$d"
  [ -s "$d/$(basename "$f")" ] && return 0
  # Kaggle rate-limits per-file downloads (HTTP 429), so back off: 1, 2, 4, 8, 16 min.
  for wait in 60 120 240 480 960; do
    err=$("$KAGGLE" competitions download gemma-4-developer-agent -f "$f" -p "$d" -q 2>&1 >/dev/null)
    [ -s "$d/$(basename "$f")" ] && { echo "ok $f"; return 0; }
    echo "retry $f in ${wait}s: ${err:0:120}"
    sleep "$wait"
  done
  echo "FAILED $f"
}
export -f dl_one; export KAGGLE
xargs -P "$PAR" -I{} bash -c 'dl_one "$@"' _ {} < "$LIST"
