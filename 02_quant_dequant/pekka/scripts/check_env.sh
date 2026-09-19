#!/usr/bin/env bash
set -euo pipefail

echo "== Host =="
uname -a

echo
echo "== CUDA compiler =="
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo "nvcc: NOT FOUND"
fi

echo
echo "== GPU and driver =="
if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total \
        --format=csv,noheader; then
        echo "WARNING: nvidia-smi exists but GPU/NVML access is unavailable."
    fi
else
    echo "nvidia-smi: NOT FOUND"
fi

echo
echo "== Build tools =="
for tool in cmake c++ python3 clang-format; do
    if command -v "${tool}" >/dev/null 2>&1; then
        printf '%-18s %s\n' "${tool}" "$(command -v "${tool}")"
        "${tool}" --version 2>/dev/null | head -n 1 || true
    else
        printf '%-18s %s\n' "${tool}" "NOT FOUND"
    fi
done

echo
echo "== CUDA analysis tools =="
for tool in compute-sanitizer ncu nsys; do
    if command -v "${tool}" >/dev/null 2>&1; then
        printf '%-18s %s\n' "${tool}" "$(command -v "${tool}")"
    else
        printf '%-18s %s\n' "${tool}" "NOT FOUND (optional)"
    fi
done
