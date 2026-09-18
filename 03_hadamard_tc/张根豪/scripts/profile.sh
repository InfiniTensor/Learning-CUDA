#!/usr/bin/env bash
# profile.sh — 为 Hadamard baseline 采集可复现的 Nsight Systems/Compute 报告。
#
# 默认代表配置：FP16、head_dim=128、16384 tokens。这个规模足以让 kernel
# 脱离纯 launch-latency 区间，同时不会生成过大的 profiler 文件。
#
# 用法：
#   bash scripts/profile.sh             # 依次尝试 nsys 和 ncu
#   bash scripts/profile.sh nsys        # 只采集系统时间线
#   bash scripts/profile.sh ncu         # 只采集 kernel 硬件指标
#
# ncu 读取 GPU 性能计数器可能需要管理员权限。脚本不会修改系统权限；失败时会
# 保留明确提示，管理员配置方法见 docs/final_report.md 的 Profiler 一节。

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/hadamard_bench"
RESULTS="$ROOT/results"
MODE="${1:-all}"
STEM="fp16_hd128_baseline"

if [[ "$MODE" != "all" && "$MODE" != "nsys" && "$MODE" != "ncu" ]]; then
  echo "Usage: $0 [all|nsys|ncu]" >&2
  exit 2
fi

if [[ ! -x "$BIN" ]]; then
  make -C "$ROOT"
fi
mkdir -p "$RESULTS"

COMMON_ARGS=(
  --batch 2 --seq 512 --heads 16 --head_dim 128
  --dtype fp16 --normalize true --warmup 5 --iters 20 --check false
)

run_nsys() {
  if ! command -v nsys >/dev/null 2>&1; then
    echo "[profile] nsys 不可用，跳过 Nsight Systems。" >&2
    return 0
  fi

  local output="$RESULTS/nsys_$STEM"
  echo "[profile] 采集 Nsight Systems: $output"
  nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --force-overwrite true -o "$output" "$BIN" "${COMMON_ARGS[@]}" || true

  # 某些 target-only 安装只生成 qdstrm。优先用 nsys import；Ubuntu 旧版
  # 包若没有暴露 import 子命令，则调用同版本的 QdstrmImporter。
  local rep="$output.nsys-rep"
  local stream="$output.qdstrm"
  # 每次都从本轮新 trace 强制刷新 rep，避免误读上一次采集的报告。
  if [[ -f "$stream" ]]; then
    if nsys import --help >/dev/null 2>&1; then
      nsys import --force-overwrite=true --output-file="$rep" "$stream" || true
    elif [[ -x /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter ]]; then
      /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter \
        -f -i "$stream" -o "$rep" || true
    fi
  fi

  if [[ -f "$rep" ]]; then
    nsys stats --force-export=true --report gpukernsum,cudaapisum \
      --timeunit usec "$rep" || true
    echo "[profile] Nsight Systems 报告: $rep"
  else
    echo "[profile] 未生成 .nsys-rep；请安装与采集端同版本的 host importer。" >&2
  fi
}

run_ncu() {
  if ! command -v ncu >/dev/null 2>&1; then
    echo "[profile] ncu 不可用，跳过 Nsight Compute。" >&2
    return 0
  fi

  local output="$RESULTS/ncu_$STEM"
  echo "[profile] 采集 Nsight Compute: $output"
  # bench 在正式计时前有一次分发校验，因此跳过第一个 kernel，只分析第二个。
  ncu --set full --launch-skip 1 --launch-count 1 --force-overwrite \
    -o "$output" "$BIN" "${COMMON_ARGS[@]}" || true

  if [[ -f "$output.ncu-rep" ]]; then
    echo "[profile] Nsight Compute 报告: $output.ncu-rep"
  else
    echo "[profile] 未生成 .ncu-rep。若出现 ERR_NVGPUCTRPERM，请让管理员开放 GPU 性能计数器。" >&2
  fi
}

if [[ "$MODE" == "all" || "$MODE" == "nsys" ]]; then
  run_nsys
fi
if [[ "$MODE" == "all" || "$MODE" == "ncu" ]]; then
  run_ncu
fi
