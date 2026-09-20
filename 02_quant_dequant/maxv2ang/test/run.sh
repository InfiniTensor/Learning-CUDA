#!/usr/bin/env bash
# ============================================================================
# nsys / ncu 包装脚本
#
# 测试主体是 C++ 程序 test/test_all（由 `make test` 先跑），本脚本只做一件事：
# 调用 NVIDIA 的两个外部分析工具，把原始输出落到 test/out/ 供写报告用。
# 正确性、误差、压缩率、kernel 计时等全部由 C++ 侧产出，不经这里。
#
# 用法（一般由 `make test` 自动调用）：
#   bash test/run.sh                  # nsys + ncu，工具缺失或无权就跳过
#   NO_PROFILING=1 bash test/run.sh   # 全部跳过
#
# 产出：
#   test/out/nsys/kern_sum.txt       各 kernel 总耗时排序
#   test/out/nsys/mem_time_sum.txt   H2D/D2H 拷贝总时长
#   test/out/nsys/prof.nsys-rep      原始时间线（可拷回本地用 GUI 打开）
#   test/out/ncu/<kernel>.csv        每个 kernel 的 --set basic 指标
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1        # 回到工程根

OUT="${OUTDIR:-test/out}"
NO_PROFILING="${NO_PROFILING:-0}"

note() { printf '  %s\n' "$*"; }
hr()   { printf '\n\033[1m==== %s ====\033[0m\n' "$*"; }
ok()   { printf '\n\033[32m✔ %s\033[0m\n' "$*"; }

# 优先用 /usr/local/cuda/bin（很多机器没把 CUDA 加进 PATH），找不到再回落 PATH
pick_tool() {
    for c in "/usr/local/cuda/bin/$1" "$(command -v "$1" 2>/dev/null)"; do
        [ -n "$c" ] && [ -x "$c" ] && { printf '%s' "$c"; return 0; }
    done
    return 1
}
NSYS="$(pick_tool nsys || true)"
NCU="$(pick_tool ncu || true)"

if [[ "$NO_PROFILING" == "1" ]]; then
    note "NO_PROFILING=1，跳过 nsys / ncu"
    exit 0
fi

if [[ ! -x ./test/test_all ]]; then
    note "找不到 ./test/test_all，先 make（本脚本不做构建）"
    exit 1
fi

mkdir -p "$OUT"

# 被 profile 的就是测试程序本身。用 --perf-only 只跑性能档（自测 + 1024² 六组），
# 避免 nsys/ncu 把 48 组覆盖档重复跑好几遍 —— ncu 每个 kernel 一次，代价很高。
PROG=(./test/test_all --perf-only)

# ---------------------------------------------------------------- nsys
if [ -n "$NSYS" ]; then
    hr "nsys：全流程时间线"
    mkdir -p "$OUT/nsys"
    if "$NSYS" profile -o "$OUT/nsys/prof" --force-overwrite true \
           "${PROG[@]}" --outdir "$OUT/nsys/run" >"$OUT/nsys/prof.log" 2>&1; then
        "$NSYS" stats "$OUT/nsys/prof.nsys-rep" --report cuda_gpu_kern_sum \
            >"$OUT/nsys/kern_sum.txt" 2>&1
        "$NSYS" stats "$OUT/nsys/prof.nsys-rep" --report cuda_gpu_mem_time_sum \
            >"$OUT/nsys/mem_time_sum.txt" 2>&1
        note "各 kernel 耗时排序："
        sed 's/^/    /' "$OUT/nsys/kern_sum.txt"
        ok "nsys 完成 → $OUT/nsys/"
    else
        note "nsys 运行失败，详见 $OUT/nsys/prof.log"
    fi
else
    note "未找到 nsys（已试 /usr/local/cuda/bin/nsys 与 PATH），跳过"
fi

# ---------------------------------------------------------------- ncu
if [ -n "$NCU" ]; then
    hr "ncu：kernel 深挖"
    mkdir -p "$OUT/ncu"
    got=0
    for k in k_nvfp4_dequant k_mxfp8_dequant k_nvfp4_quant_block k_mxfp8_quant_block; do
        # -k 只profile匹配的 kernel；--launch-skip 1 跳过首个（warmup），只取一个
        if "$NCU" --set basic -k "regex:$k" --launch-skip 1 --launch-count 1 --csv \
               "${PROG[@]}" --outdir "$OUT/ncu/run" >"$OUT/ncu/$k.csv" 2>&1; then
            note "$k.csv 已生成"
            got=1
        else
            note "$k 采集失败 —— 最常见原因是无 profiling 权限（ERR_NVGPUCTRPERM）；"
            note "  需管理员放开 NVreg_RestrictProfilingToAdminUsers=0，或容器加 CAP_SYS_ADMIN"
            break
        fi
    done
    [[ $got -eq 1 ]] && ok "ncu 完成 → $OUT/ncu/" || note "ncu 无可上传数据"
else
    note "未找到 ncu（已试 /usr/local/cuda/bin/ncu 与 PATH），跳过"
fi

note "提示：ncu 会锁频并多次重放 kernel，其时间不能当作性能数据上报；"
note "      性能数字一律取自 test/test_all 的 cudaEvent 计时（重复取最小）。"

