#!/usr/bin/env bash
set -euo pipefail
umask 077
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
console_python="${T1_CONSOLE_PYTHON:-$HOME/miniforge3/envs/cuda-dev/bin/python}"
if ! console_python="$(command -v "$console_python")"; then
  echo 'Missing cuda-dev Python: create the conda environment or set T1_CONSOLE_PYTHON to an existing interpreter.' >&2
  exit 1
fi
# Keep private launcher configuration from selecting a different interpreter.
export T1_CONSOLE_PYTHON="$console_python"
exec "$console_python" -B "$repo_root/scripts/vehicle_console_launcher.py" "$@"
