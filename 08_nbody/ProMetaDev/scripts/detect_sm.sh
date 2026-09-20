#!/usr/bin/env bash
# 输出逗号分隔的 CUDA SM 代号列表 (供 CMake 循环 -gencode 用)
#  策略:
#   1) 优先 nvidia-smi --query-gpu=compute_cap → 官方标准 (精确)
#   2) 回退: 用 nvidia-smi gpu name 查表 (heuristic)
#   3) 再回退: 输出默认列表 "75,80,86,89,90" (覆盖 Turing~Blackwell 全部常见)
#  最后总会输出至少一行，保证 CMake 可继续
set -u

# -------- helper: 输出一行 (stdout) ----------
SM_LIST=""

# --- 策略 1: nvidia-smi compute_cap ---
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')
if [ -n "$CAP" ]; then
    MAJ="${CAP%%.*}"
    MIN="${CAP#*.}"
    if [ "$MAJ" -ge 1 ] && [ -n "$MIN" ] 2>/dev/null; then
        SM=$(( MAJ*10 + MIN ))
        # 已知 nvcc <12.3 还没 sm_120 等 >100 的 token; 这时降级到 90 + compute_90 JIT
        if [ "$SM" -gt 100 ]; then
            echo "detect_sm: 检测到 SM=$SM (Blackwell/ultra-new) → 编译为 sm_90 + PTX JIT" 1>&2
            SM_LIST="90"
        else
            SM_LIST="$SM"
        fi
    fi
fi

# --- 策略 2: name 查表 ---
if [ -z "$SM_LIST" ]; then
    NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
    if [ -n "$NAME" ]; then
        case "$NAME" in
            *v100*|*titanv*)                  SM_LIST="70" ;;
            *rtx20*|*rtx16*|*t1000*|*t600*)  SM_LIST="75" ;;
            *a100*|*a30*)                    SM_LIST="80" ;;
            *a10*|*a40*|*a5000*|*a6000*|*rtx30*) SM_LIST="86" ;;
            *l40*|*rtx40*)                   SM_LIST="89" ;;
            *rtx50*|*h100*|*b100*|*b200*)    SM_LIST="90" ;;
        esac
        [ -n "$SM_LIST" ] && echo "detect_sm: 按 GPU 名查表 -> $SM_LIST" 1>&2
    fi
fi

# --- 策略 3: fallback 默认 (跨 GPU 通用多码片) ---
if [ -z "$SM_LIST" ]; then
    SM_LIST="75,80,86,89,90"
    echo "detect_sm: nvidia-smi 不可用 → 采用 fallback SM 列表 $SM_LIST" 1>&2
fi

echo "$SM_LIST"
