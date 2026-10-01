#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/requirements" ]]; then
    DEFAULT_RELEASE_ROOT="${SCRIPT_DIR}"
else
    DEFAULT_RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
RELEASE_ROOT="${1:-${DEFAULT_RELEASE_ROOT}}"
WHEEL_PYTHON="${UNICELL_WHEEL_PYTHON:-python}"

case "$(uname -m)" in
    aarch64|arm64)
        PLATFORM="linux-aarch64"
        ;;
    x86_64|amd64)
        PLATFORM="linux-x86_64"
        ;;
    *)
        echo "Unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

PYTHON_VERSION="$(${WHEEL_PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.9" ]]; then
    echo "Wheelhouse must be built with Python 3.9, got ${PYTHON_VERSION}." >&2
    exit 1
fi

REQUIREMENTS="${RELEASE_ROOT}/requirements/${PLATFORM}.txt"
WHEELHOUSE="${RELEASE_ROOT}/wheelhouse/${PLATFORM}"

if [[ ! -f "${REQUIREMENTS}" ]]; then
    echo "Missing requirements file: ${REQUIREMENTS}" >&2
    exit 1
fi

mkdir -p "${WHEELHOUSE}"
PIP_ARGS=(
    --wheel-dir "${WHEELHOUSE}"
    --find-links "${WHEELHOUSE}"
    --requirement "${REQUIREMENTS}"
)

if [[ "${PLATFORM}" == "linux-x86_64" ]]; then
    PIP_ARGS+=(--extra-index-url https://download.pytorch.org/whl/cu117)
fi

"${WHEEL_PYTHON}" -m pip wheel --no-cache-dir "${PIP_ARGS[@]}"

if find "${WHEELHOUSE}" -maxdepth 1 -type f \( -name '*.tar.gz' -o -name '*.zip' \) | grep -q .; then
    echo "Source archives remain in ${WHEELHOUSE}; build them into wheels before offline release." >&2
    exit 1
fi

echo "Wheelhouse ready: ${WHEELHOUSE}"
