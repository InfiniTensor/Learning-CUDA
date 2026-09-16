#!/usr/bin/env bash
# Collect hardware/software versions without installing or changing the environment.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-output/results/environment.txt}"
PLATFORM="${PLATFORM:-nvidia}"
COREX_HOME="${COREX_HOME:-/usr/local/corex}"
MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
MACA_HOME="${MACA_HOME:-/opt/maca}"
MACA_CUDA="${MACA_CUDA:-${MACA_HOME}/tools/cu-bridge}"
mkdir -p "$(dirname "${OUT}")"

{
    echo "collected_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname=$(hostname)"
    echo "os=$(uname -a)"
    if test -f /etc/os-release; then
        grep -E '^(NAME|VERSION|VERSION_ID)=' /etc/os-release
    fi
    echo
    echo "[gpu]"
    if [[ "${PLATFORM}" == "iluvatar" ]]; then
        if command -v ixsmi >/dev/null 2>&1; then
            ixsmi 2>&1 || true
        else
            echo "ixsmi=not-found"
        fi
    elif [[ "${PLATFORM}" == "moore" ]]; then
        MUSA_INFO="${MUSA_HOME}/bin/musaInfo"
        if test -x "${MUSA_INFO}"; then
            LD_LIBRARY_PATH="${MUSA_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
                "${MUSA_INFO}"
        else
            echo "musaInfo=not-found (${MUSA_INFO})"
        fi
    elif [[ "${PLATFORM}" == "metax" ]]; then
        if command -v mx-smi >/dev/null 2>&1; then
            mx-smi --version 2>&1 || true
            mx-smi 2>&1 || true
        else
            echo "mx-smi=not-found"
        fi
    elif command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap \
            --format=csv,noheader
    else
        echo "gpu-query=not-found for PLATFORM=${PLATFORM}"
    fi
    echo
    echo "[cpu]"
    echo "nproc=$(nproc)"
    if command -v lscpu >/dev/null 2>&1; then
        lscpu | grep -E '^(Architecture|CPU\(s\)|Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)):' || true
    fi
    echo
    echo "[accelerator-toolkit]"
    if [[ "${PLATFORM}" == "iluvatar" ]]; then
        COREX_COMPILER="${COREX_CXX:-${COREX_HOME}/bin/clang++}"
        if test -x "${COREX_COMPILER}"; then
            echo "corex-clang++=${COREX_COMPILER}"
            "${COREX_COMPILER}" --version
        else
            echo "corex-clang++=not-found (${COREX_COMPILER})"
        fi
        COREX_NVCC="${COREX_NVCC:-${COREX_HOME}/bin/nvcc}"
        if test -x "${COREX_NVCC}"; then
            echo "corex-nvcc=${COREX_NVCC}"
            "${COREX_NVCC}" --version
        else
            echo "corex-nvcc=not-found (${COREX_NVCC})"
        fi
        echo "COREX_HOME=${COREX_HOME}"
        test -L "${COREX_HOME}" && echo "COREX_HOME_REALPATH=$(readlink -f "${COREX_HOME}")"
    elif [[ "${PLATFORM}" == "moore" ]]; then
        MUSA_COMPILER="${MCC:-${MUSA_HOME}/bin/mcc}"
        if test -x "${MUSA_COMPILER}"; then
            echo "mcc=${MUSA_COMPILER}"
            "${MUSA_COMPILER}" --version
            if test -x "${MUSA_HOME}/bin/musa_version_query"; then
                "${MUSA_HOME}/bin/musa_version_query"
            fi
        else
            echo "mcc=not-found (${MUSA_COMPILER})"
        fi
    elif [[ "${PLATFORM}" == "metax" ]]; then
        MACA_COMPILER="${MXCC:-${MACA_HOME}/mxgpu_llvm/bin/mxcc}"
        if test -x "${MACA_COMPILER}"; then
            echo "mxcc=${MACA_COMPILER}"
            "${MACA_COMPILER}" --version
        else
            echo "mxcc=not-found (${MACA_COMPILER})"
        fi
        echo "MACA_HOME=${MACA_HOME}"
        echo "MACA_CUDA=${MACA_CUDA}"
        test -L "${MACA_HOME}" && echo "MACA_HOME_REALPATH=$(readlink -f "${MACA_HOME}")"
    else
        CUDA_COMPILER="${NVCC:-nvcc}"
        if command -v "${CUDA_COMPILER}" >/dev/null 2>&1 || test -x "${CUDA_COMPILER}"; then
            echo "nvcc=${CUDA_COMPILER}"
            "${CUDA_COMPILER}" --version
        else
            echo "nvcc=not-found"
        fi
    fi
    echo
    echo "[compiler]"
    "${CXX_HOST:-g++}" --version | head -n 1
    make --version | head -n 1
    echo
    echo "[profilers]"
    if command -v nsys >/dev/null 2>&1; then nsys --version; else echo "nsys=not-found"; fi
    if command -v ncu >/dev/null 2>&1; then ncu --version; else echo "ncu=not-found"; fi
    if command -v muprof >/dev/null 2>&1; then muprof --version; else echo "muprof=not-found"; fi
    if command -v mxprof >/dev/null 2>&1; then mxprof --version; else echo "mxprof=not-found"; fi
    if command -v ixprof >/dev/null 2>&1; then ixprof --version; else echo "ixprof=not-found"; fi
    if test -x "${MACA_HOME}/bin/mcTracer"; then
        echo "mcTracer=${MACA_HOME}/bin/mcTracer"
        "${MACA_HOME}/bin/mcTracer" --help 2>&1 | grep -A 1 -m 1 'Version:' || true
    else
        echo "mcTracer=not-found"
    fi
    echo
    echo "[build-contract]"
    echo "PLATFORM=${PLATFORM}"
    if [[ "${PLATFORM}" == "nvidia" ]]; then
        echo "ARCH=${ARCH:-compiler-default}"
    elif [[ "${PLATFORM}" == "iluvatar" ]]; then
        echo "COREX_HOME=${COREX_HOME}"
        echo "COREX_CXX=${COREX_CXX:-${COREX_HOME}/bin/clang++}"
    elif [[ "${PLATFORM}" == "metax" ]]; then
        echo "MACA_HOME=${MACA_HOME}"
        echo "MACA_CUDA=${MACA_CUDA}"
        echo "MXCC=${MXCC:-${MACA_HOME}/mxgpu_llvm/bin/mxcc}"
    else
        echo "MUSA_HOME=${MUSA_HOME}"
    fi
    echo "FAST_EXP=${FAST_EXP:-0}"
    echo "WITH_OPENCV=${WITH_OPENCV:-0}"
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "git_commit=$(git rev-parse HEAD)"
        if test -n "$(git status --porcelain)"; then
            echo "git_worktree=dirty"
        else
            echo "git_worktree=clean"
        fi
    else
        echo "git_commit=unavailable"
        echo "git_worktree=unavailable (uploaded source snapshot)"
    fi
} | tee "${OUT}"

echo "Environment manifest: ${OUT}"
