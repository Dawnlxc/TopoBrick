#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPOSITORY_ROOT"

exec "${TOPOBRICK_PYTHON:-python}" -m topobrick.run \
  --config configs/bts_site_b.yaml \
  "$@"
