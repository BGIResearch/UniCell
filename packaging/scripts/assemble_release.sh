#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="0.2.1"
RELEASE_PARENT="${UNICELL_RELEASE_OUT:-${REPO_ROOT}/dist/releases}"
RELEASE_DIR="${RELEASE_PARENT}/unicell_fig5-${VERSION}"
ARCHIVE="${RELEASE_PARENT}/unicell_fig5-${VERSION}-multiarch-bootstrap.tar.gz"
TEMP_ARCHIVE="${TMPDIR:-/tmp}/unicell_fig5-${VERSION}-multiarch-bootstrap.$$.tar.gz"
APP_WHEEL="${REPO_ROOT}/dist/unicell_fig5-${VERSION}-py3-none-any.whl"
ARM_WHEEL_DIR="${UNICELL_ARM_TORCH_WHEEL_DIR:-}"

if [[ -z "${ARM_WHEEL_DIR}" ]]; then
    echo "Set UNICELL_ARM_TORCH_WHEEL_DIR to the directory containing the ARM PyTorch wheels." >&2
    exit 1
fi

if [[ ! -f "${APP_WHEEL}" ]]; then
    echo "Build the application wheel first: bash packaging/scripts/build_wheel.sh" >&2
    exit 1
fi
if [[ -e "${RELEASE_DIR}" || -e "${ARCHIVE}" ]]; then
    echo "Release output already exists. Move it aside before rebuilding:" >&2
    echo "  ${RELEASE_DIR}" >&2
    echo "  ${ARCHIVE}" >&2
    exit 1
fi

mkdir -p \
    "${RELEASE_DIR}/common/models/last_version_gml" \
    "${RELEASE_DIR}/requirements" \
    "${RELEASE_DIR}/wheelhouse/linux-aarch64" \
    "${RELEASE_DIR}/wheelhouse/linux-x86_64"

cp "${APP_WHEEL}" "${RELEASE_DIR}/common/"
cp "${REPO_ROOT}/safe_list.json" "${RELEASE_DIR}/common/"

MODEL_FILES=(
    unicell_v1.best.pth
    gene_names.pk
    ontoGraph.pk
    ontoGraph.graph.gml
    celltype_dict.pk
    tissue_dict.pk
    species_dict.pk
)
for model_file in "${MODEL_FILES[@]}"; do
    cp \
        "${REPO_ROOT}/models/last_version_gml/${model_file}" \
        "${RELEASE_DIR}/common/models/last_version_gml/"
done

cp "${REPO_ROOT}/packaging/requirements/"*.txt "${RELEASE_DIR}/requirements/"
cp "${REPO_ROOT}/packaging/scripts/install.sh" "${RELEASE_DIR}/install.sh"
cp "${REPO_ROOT}/packaging/scripts/populate_wheelhouse.sh" "${RELEASE_DIR}/populate_wheelhouse.sh"
cp "${REPO_ROOT}/packaging/scripts/smoke_test.py" "${RELEASE_DIR}/smoke_test.py"
cp "${REPO_ROOT}/packaging/README.md" "${RELEASE_DIR}/README.md"
cp "${REPO_ROOT}/packaging/USAGE.md" "${RELEASE_DIR}/USAGE.md"
cp "${REPO_ROOT}/packaging/USAGE_ZH.md" "${RELEASE_DIR}/USAGE_ZH.md"
cp "${REPO_ROOT}/LICENSE" "${RELEASE_DIR}/LICENSE"
cp "${REPO_ROOT}/THIRD_PARTY_NOTICES.md" "${RELEASE_DIR}/THIRD_PARTY_NOTICES.md"
cp -R "${REPO_ROOT}/licenses" "${RELEASE_DIR}/licenses"

ARM_TORCH_WHEEL="${ARM_WHEEL_DIR}/torch-2.0.0+cuda11.6.gcc9.3-cp39-cp39-linux_aarch64.whl"
ARM_TORCHTEXT_WHEEL="${ARM_WHEEL_DIR}/torchtext-0.15.2a0+4571036-cp39-cp39-linux_aarch64.whl"
for platform_wheel in "${ARM_TORCH_WHEEL}" "${ARM_TORCHTEXT_WHEEL}"; do
    if [[ ! -f "${platform_wheel}" ]]; then
        echo "Missing required ARM wheel: ${platform_wheel}" >&2
        exit 1
    fi
done

# The historical cluster wheel filename contains an extra `.gcc9.3` local-version
# component which is absent from its internal METADATA. Normalize only the outer
# filename so pip can resolve `torch==2.0.0+cuda11.6`; the wheel bytes are unchanged.
cp \
    "${ARM_TORCH_WHEEL}" \
    "${RELEASE_DIR}/wheelhouse/linux-aarch64/torch-2.0.0+cuda11.6-cp39-cp39-linux_aarch64.whl"
cp "${ARM_TORCHTEXT_WHEEL}" "${RELEASE_DIR}/wheelhouse/linux-aarch64/"

(
    cd "${RELEASE_DIR}"
    find . -type f ! -name SHA256SUMS -print0 \
        | sort -z \
        | xargs -0 sha256sum > SHA256SUMS
)

# Some shared filesystems update directory metadata while tar traverses them.
# Build in local temporary storage and suppress only that metadata warning.
tar \
    --warning=no-file-changed \
    --ignore-failed-read \
    -C "${RELEASE_PARENT}" \
    -czf "${TEMP_ARCHIVE}" \
    "$(basename "${RELEASE_DIR}")"
gzip -t "${TEMP_ARCHIVE}"
mv "${TEMP_ARCHIVE}" "${ARCHIVE}"

echo "Release directory: ${RELEASE_DIR}"
echo "Release archive:   ${ARCHIVE}"
echo "ARM bootstrap wheels are included."
echo "Populate and test linux-x86_64/wheelhouse on a native x86_64 node."
