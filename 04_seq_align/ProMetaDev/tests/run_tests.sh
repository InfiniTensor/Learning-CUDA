#!/bin/bash
# 端到端测试：自检 / GPU 与 CPU 穷举逐行比对 / 真值召回 / 随机 read 拒绝
#   用法: bash tests/run_tests.sh [SM_ARCH]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SM_ARCH="${1:-90}"
BIN="$ROOT/build/seqalign"
GEN="$ROOT/tools/gen_data.py"
TMP="$ROOT/tests/tmp"
mkdir -p "$TMP"
cd "$ROOT"

fail=0
step() { echo ""; echo "=========================================="; echo "$1"; echo "=========================================="; }
ok()   { echo "[PASS] $1"; }
bad()  { echo "[FAIL] $1"; fail=1; }

# ---------------------------------------------------------------- 0. 构建
step "[0/4] 构建 (sm_${SM_ARCH})"
bash build.sh "$SM_ARCH" >/dev/null 2>&1
ok "构建完成"

# ---------------------------------------------------------------- 1. 自检
step "[1/4] 引擎自检"
if "$BIN" selftest; then ok "selftest"; else bad "selftest"; fi

# ---------------------------------------------------------------- 2. GPU vs CPU 穷举
step "[2/4] GPU 与 CPU 穷举逐行比对（小规模）"
# 用小参考（CPU 穷举代价正比于参考长度），覆盖多组 (k, band)
run_parity() {
    local name="$1" k="$2" band="$3" step_seed="$4"
    local ref="$TMP/${name}.fa" rd="$TMP/${name}.fq"
    python3 "$GEN" --ref-seqs 3 --ref-len 5000 --reads 40 --read-len 120 \
        --sub-rate 0.03 --random-frac 0.25 --seed 42 \
        --out-ref "$ref" --out-reads "$rd" --out-truth "$TMP/${name}.truth" >/dev/null
    "$BIN" cpu "$ref" "$rd" "$TMP/${name}.cpu.txt" --k "$k" --band "$band" >/dev/null
    "$BIN" gpu "$ref" "$rd" "$TMP/${name}.gpu.txt" --k "$k" --band "$band" \
        --seed-step "$step_seed" >/dev/null
    if diff -q "$TMP/${name}.cpu.txt" "$TMP/${name}.gpu.txt" >/dev/null; then
        ok "k=$k band=$band step=$step_seed : GPU 与 CPU 穷举完全一致（40/40）"
    else
        bad "k=$k band=$band step=$step_seed : 结果不一致"
        diff "$TMP/${name}.cpu.txt" "$TMP/${name}.gpu.txt" | head -10
    fi
}

run_parity "p_k11b4"  11  4  4
run_parity "p_k13b8"  13  8  8
run_parity "p_k15b8"  15  8  8
run_parity "p_k15b16" 15 16 12

# ---------------------------------------------------------------- 3. 真值召回
step "[3/4] 真值召回与随机 read 拒绝（中等规模）"
MREF="$TMP/m.fa"; MRD="$TMP/m.fq"; MTR="$TMP/m.truth"
if [ ! -f "$MREF" ]; then
    python3 "$GEN" --ref-seqs 8 --ref-len 200000 --reads 2000 --read-len 150 \
        --sub-rate 0.03 --random-frac 0.2 --seed 7 \
        --out-ref "$MREF" --out-reads "$MRD" --out-truth "$MTR" >/dev/null
fi
out=$("$BIN" gpu "$MREF" "$MRD" "$TMP/m.gpu.txt" --k 15 --band 8 --seed-step 8 \
        --truth "$MTR" 2>&1)
echo "$out" | grep -E '^\[gpu\]|^\[check\]' || true
aligned=$(echo "$out" | grep "有来源 read" | sed -n 's/.*: \([0-9]*\)\/\([0-9]*\) 被比对.*/\1 \2/p')
unknown=$(echo "$out" | grep "随机 read" | sed -n 's/.*: \([0-9]*\)\/\([0-9]*\) 正确.*/\1 \2/p')
a1=$(echo "$aligned" | cut -d' ' -f1); a2=$(echo "$aligned" | cut -d' ' -f2)
u1=$(echo "$unknown" | cut -d' ' -f1); u2=$(echo "$unknown" | cut -d' ' -f2)
[ "$a1" = "$a2" ] && ok "有来源 read 全部被比对 ($a1/$a2)" || bad "有来源 read 漏比对 ($a1/$a2)"
[ "$u1" = "$u2" ] && ok "随机 read 全部判为 unknown_origin ($u1/$u2)" || bad "随机 read 误判 ($u1/$u2)"

# ---------------------------------------------------------------- 4. 查询无关性/确定性
step "[4/4] 结果确定性（同一输入重复运行结果一致）"
"$BIN" gpu "$MREF" "$MRD" "$TMP/m.run2.txt" --k 15 --band 8 --seed-step 8 >/dev/null
if diff -q "$TMP/m.gpu.txt" "$TMP/m.run2.txt" >/dev/null; then
    ok "重复运行结果逐字节一致"
else
    bad "重复运行结果不一致"
fi

echo ""
if [ "$fail" -eq 0 ]; then
    echo "=========================================="; echo "全部测试通过 ✓"; echo "=========================================="
else
    echo "=========================================="; echo "存在失败测试 ✗"; echo "=========================================="
    exit 1
fi
