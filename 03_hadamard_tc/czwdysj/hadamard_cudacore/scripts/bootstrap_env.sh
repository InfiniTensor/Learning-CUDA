#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
DAO_DIR="${PROJECT_DIR}/third_party/fast-hadamard-transform"
DAO_COMMIT="e7706faf8d1c3b9f241e36860640ad1dac644ede"
DAO_MARKER="${VENV_DIR}/.dao_fht_commit"
PYTORCH_VERSION="2.9.1+cu130"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[bootstrap] Creating ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi

PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

"${PIP}" install --upgrade pip setuptools wheel
"${PIP}" install --upgrade ninja packaging numpy

if ! "${PYTHON}" -c \
    "import torch; assert torch.__version__ == '${PYTORCH_VERSION}'" \
    >/dev/null 2>&1; then
    echo "[bootstrap] Installing torch==${PYTORCH_VERSION}"
    "${PIP}" install --index-url https://download.pytorch.org/whl/cu130 \
        "torch==${PYTORCH_VERSION}"
else
    echo "[bootstrap] PyTorch ${PYTORCH_VERSION} already installed"
fi

mkdir -p "${PROJECT_DIR}/third_party"
if [[ ! -d "${DAO_DIR}/.git" ]]; then
    echo "[bootstrap] Cloning Dao-AILab/fast-hadamard-transform"
    git clone https://github.com/Dao-AILab/fast-hadamard-transform.git "${DAO_DIR}"
fi

if ! git -C "${DAO_DIR}" diff --quiet || \
   ! git -C "${DAO_DIR}" diff --cached --quiet; then
    echo "[bootstrap] Refusing to overwrite local Dao source changes in ${DAO_DIR}" >&2
    exit 1
fi
git -C "${DAO_DIR}" fetch origin "${DAO_COMMIT}" --depth 1
git -C "${DAO_DIR}" checkout --detach "${DAO_COMMIT}"

if "${PYTHON}" -c "from fast_hadamard_transform import hadamard_transform" \
    >/dev/null 2>&1 && [[ -f "${DAO_MARKER}" ]] && \
    grep -qx "${DAO_COMMIT}" "${DAO_MARKER}"; then
    echo "[bootstrap] Dao FHT Python extension already importable"
else
    echo "[bootstrap] Building Dao FHT at ${DAO_COMMIT} for SM 12.0"
    (
        cd "${DAO_DIR}"
        PATH="${VENV_DIR}/bin:${PATH}" \
        CUDA_HOME=/usr/local/cuda \
        TORCH_CUDA_ARCH_LIST=12.0 \
        MAX_JOBS="${MAX_JOBS:-4}" \
        "${PIP}" install -v --no-build-isolation --no-deps .
    )
    printf '%s\n' "${DAO_COMMIT}" > "${DAO_MARKER}"
fi

echo "[bootstrap] Verifying GPU, PyTorch, Dao import and one normalized launch"
"${PYTHON}" - <<'PY'
import math
import torch
from fast_hadamard_transform import hadamard_transform

if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is unavailable")
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
if capability != (12, 0):
    raise SystemExit(f"Expected SM 12.0, found SM {capability[0]}.{capability[1]}")
x = torch.arange(256, device="cuda", dtype=torch.float16).reshape(2, 128) / 128
y = hadamard_transform(x, scale=1.0 / math.sqrt(128))
torch.cuda.synchronize()
if y.shape != x.shape or not torch.isfinite(y).all():
    raise SystemExit("Dao FHT verification launch returned an invalid tensor")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"gpu={name}")
print(f"compute_capability={capability[0]}.{capability[1]}")
print("dao_fht_import_and_launch=PASS")
PY

echo "[bootstrap] Environment ready: ${VENV_DIR}"
