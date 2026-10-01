#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/build_wheel.sh"
bash "${SCRIPT_DIR}/assemble_release.sh"

