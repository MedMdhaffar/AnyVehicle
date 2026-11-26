#!/usr/bin/env bash
# WAYCON Vehicle Services — one RTSP camera, headless, foreground.
#
# Camera (exactly one; the URL is never printed, logged or put on argv):
#   WAYCON_RTSP_URI_FILE=/path/to/camera.uri   owned mode-0600 file with the rtsp:// URL (recommended)
#   WAYCON_RTSP_URI=rtsp://...                 alternative; removed from the environment before any child starts
# Runtime (defaults from ~/.config/waycon/console.json when present):
#   T1_CONSOLE_PYTHON   interpreter (default: <repo>/.venv/bin/python)
#   T1_REDIS_SERVER     redis-server binary for the run-owned Redis
#   WAYCON_POSTGRES_MODE owned (default) | existing   (DB1 disposable fixture; --console-only disables DB1)
#   WAYCON_RTSP_ROOT    private state directory (default: vehicle/media/waycon-rtsp)
# Usage: scripts/run_waycon_rtsp.sh [--check] [--console-only] [--transport tcp|udp] [--sample-every N]
#                                   [--max-seconds S] [--reconnect-attempts N] [--reconnect-delay S] [--max-runs N]
# Ctrl+C stops the input, drains model work and retires every owned process.
set -euo pipefail
umask 077
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
config="${WAYCON_CONFIG:-$HOME/.config/waycon/console.json}"
python="${T1_CONSOLE_PYTHON:-}"
if [[ -z "$python" && -f "$config" ]]; then
  python="$("${repo_root}/.venv/bin/python" -I -c 'import json,sys;print(json.load(open(sys.argv[1])).get("python",""))' "$config" 2>/dev/null || true)"
fi
python="${python:-$repo_root/.venv/bin/python}"
if ! python="$(command -v -- "$python")" || [[ ! -x "$python" ]]; then
  echo 'run_waycon_rtsp: Python environment not found. Set T1_CONSOLE_PYTHON to the WAYCON interpreter (cuda-dev + redis + psycopg).' >&2
  exit 2
fi
if ! "$python" -I -c 'import cv2, numpy, redis, torch' >/dev/null 2>&1; then
  echo 'run_waycon_rtsp: the selected Python lacks cv2/numpy/redis/torch. Set T1_CONSOLE_PYTHON to the WAYCON interpreter.' >&2
  exit 2
fi
export T1_CONSOLE_PYTHON="$python"
# exec: the supervisor receives Ctrl+C directly; the worker runs in its own session and
# is stopped only through the supervisor's controlled Stop.
exec "$python" -B "$repo_root/scripts/vehicle_rtsp_launcher.py" "$@"
