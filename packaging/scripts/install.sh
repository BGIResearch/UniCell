#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${1:-${RELEASE_ROOT}/runtime}"

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

find_env_manager() {
    if [[ -n "${UNICELL_ENV_MANAGER:-}" && -x "${UNICELL_ENV_MANAGER}" ]]; then
        printf '%s\n' "${UNICELL_ENV_MANAGER}"
    elif command -v mamba >/dev/null 2>&1; then
        command -v mamba
    elif command -v conda >/dev/null 2>&1; then
        command -v conda
    elif command -v micromamba >/dev/null 2>&1; then
        command -v micromamba
    else
        return 1
    fi
}

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    if [[ -n "${UNICELL_BASE_PYTHON:-}" ]]; then
        if [[ ! -x "${UNICELL_BASE_PYTHON}" ]]; then
            echo "UNICELL_BASE_PYTHON is not executable: ${UNICELL_BASE_PYTHON}" >&2
            exit 1
        fi
        BASE_PYTHON_VERSION="$(${UNICELL_BASE_PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        if [[ "${BASE_PYTHON_VERSION}" != "3.9" ]]; then
            echo "UNICELL_BASE_PYTHON must be Python 3.9, got ${BASE_PYTHON_VERSION}." >&2
            exit 1
        fi
        "${UNICELL_BASE_PYTHON}" -m venv "${ENV_PREFIX}"
    else
        if ! ENV_MANAGER="$(find_env_manager)"; then
            echo "No mamba, conda, or micromamba executable found." >&2
            echo "Set UNICELL_BASE_PYTHON=/absolute/path/to/python3.9 to use venv," >&2
            echo "or UNICELL_ENV_MANAGER=/absolute/path/to/mamba." >&2
            exit 1
        fi
        "${ENV_MANAGER}" create --yes --prefix "${ENV_PREFIX}" python=3.9.18 pip=24
    fi
fi

PYTHON_BIN="${ENV_PREFIX}/bin/python"
WHEELHOUSE="${RELEASE_ROOT}/wheelhouse/${PLATFORM}"
REQUIREMENTS="${RELEASE_ROOT}/requirements/${PLATFORM}.txt"
APP_WHEEL="${RELEASE_ROOT}/common/unicell_fig5-0.2.1-py3-none-any.whl"

for required_path in "${WHEELHOUSE}" "${REQUIREMENTS}" "${APP_WHEEL}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing release component: ${required_path}" >&2
        exit 1
    fi
done

PIP_ARGS=(--find-links "${WHEELHOUSE}")
if [[ "${UNICELL_OFFLINE:-0}" == "1" ]]; then
    PIP_ARGS+=(--no-index)
elif [[ "${PLATFORM}" == "linux-x86_64" ]]; then
    PIP_ARGS+=(--extra-index-url https://download.pytorch.org/whl/cu117)
fi

"${PYTHON_BIN}" -m pip install \
    "${PIP_ARGS[@]}" \
    --requirement "${REQUIREMENTS}"

"${PYTHON_BIN}" -m pip install --force-reinstall --no-deps "${APP_WHEEL}"
"${PYTHON_BIN}" -m pip check

if [[ "${UNICELL_SKIP_MODEL_CHECK:-0}" != "1" ]]; then
    "${PYTHON_BIN}" "${RELEASE_ROOT}/smoke_test.py" \
        --checkpoint-dir "${RELEASE_ROOT}/common/models/last_version_gml"
fi

echo "Installed UniCell 0.2.1 for ${PLATFORM}: ${ENV_PREFIX}"
echo "Activate with: source ${ENV_PREFIX}/bin/activate"
