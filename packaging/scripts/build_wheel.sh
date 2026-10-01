#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="0.2.1"
BUILD_PYTHON="${UNICELL_BUILD_PYTHON:-python3}"
OUT_DIR="${UNICELL_WHEEL_OUT:-${REPO_ROOT}/dist}"

if ! command -v "${BUILD_PYTHON}" >/dev/null 2>&1 && [[ ! -x "${BUILD_PYTHON}" ]]; then
    echo "Build Python is not executable: ${BUILD_PYTHON}" >&2
    exit 1
fi

PYTHON_VERSION="$(${BUILD_PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.9" ]]; then
    echo "UniCell release ${VERSION} must be built with Python 3.9, got ${PYTHON_VERSION}." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
cd "${REPO_ROOT}"

"${BUILD_PYTHON}" -m build \
    --wheel \
    --no-isolation \
    --outdir "${OUT_DIR}"

WHEEL="${OUT_DIR}/unicell_fig5-${VERSION}-py3-none-any.whl"
if [[ ! -f "${WHEEL}" ]]; then
    echo "Expected wheel was not produced: ${WHEEL}" >&2
    exit 1
fi

echo "Built: ${WHEEL}"
WHEEL_LIST="$("${BUILD_PYTHON}" -m zipfile -l "${WHEEL}")"
for required_member in \
    "unicell/cli/train_backbone.py" \
    "unicell/repo/geneformer/gene_vocab.json" \
    "unicell/repo/scgpt/tokenizer/default_gene_vocab.json" \
    "unicell_fig5-${VERSION}.dist-info/entry_points.txt"; do
    if ! grep -Fq "${required_member}" <<<"${WHEEL_LIST}"; then
        echo "Wheel is missing required member: ${required_member}" >&2
        exit 1
    fi
done

if grep -Eq '\.(pt|pth|bin|safetensors)([[:space:]]|$)' <<<"${WHEEL_LIST}"; then
    echo "Wheel unexpectedly contains model weights." >&2
    exit 1
fi

chmod 0644 "${WHEEL}"
echo "Verified training CLI/resources and confirmed that model weights are excluded."
