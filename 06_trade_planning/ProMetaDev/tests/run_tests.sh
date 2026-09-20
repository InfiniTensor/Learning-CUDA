#!/bin/bash
# 端到端测试脚本：自测 + 多规模正确性 + 查询独立性
set -e

SM_ARCH="${SM_ARCH:-90}"
MAXFLOW="$(dirname "$0")/../build/maxflow"
GEN="$(dirname "$0")/../tools/gen_graph.py"
TMP="$(dirname "$0")/tmp"
mkdir -p "$TMP"

echo "=========================================="
echo "[1/4] 构建项目 (sm_${SM_ARCH})"
echo "=========================================="
bash "$(dirname "$0")/../build.sh" "$SM_ARCH" >/dev/null 2>&1
echo "构建完成"

echo ""
echo "=========================================="
echo "[2/4] 自测 selftest"
echo "=========================================="
"$MAXFLOW" selftest
echo "selftest 通过"

echo ""
echo "=========================================="
echo "[3/4] 正确性测试（GPU vs CPU Edmonds-Karp，每档 12 个查询）"
echo "=========================================="
fail=0
run_case() {
    local name="$1" N="$2" M="$3"
    local g="$TMP/${name}.csr" q="$TMP/${name}.q" r="$TMP/${name}.result"
    python3 "$GEN" "$N" "$M" "$g" "$q" --seed 42 --queries 12 --type random
    echo ""
    echo "--- Case: $name (N=$N, M=$M) ---"
    if "$MAXFLOW" run "$g" "$q" "$r" --cpu-ref; then
        echo "[PASS] $name"
    else
        echo "[FAIL] $name"
        fail=1
    fi
}

run_case "tiny"    20      100
run_case "small"  100      800
run_case "medium" 500     4000
run_case "large" 2000    20000
run_case "big"  10000   100000
run_case "target" 100000 1000000

echo ""
echo "=========================================="
echo "[4/4] 查询独立性测试（打乱查询顺序）"
echo "=========================================="
N=200; M=1500
G="$TMP/indep.csr"
Q1="$TMP/indep_orig.q"
Q2="$TMP/indep_shuf.q"
python3 "$GEN" "$N" "$M" "$G" "$Q1" --seed 99 --queries 12 --type random
# 打乱查询顺序
shuf "$Q1" > "$Q2"
R1="$TMP/indep_orig.result"
R2="$TMP/indep_shuf.result"
"$MAXFLOW" run "$G" "$Q1" "$R1" >/dev/null
"$MAXFLOW" run "$G" "$Q2" "$R2" >/dev/null
# 按源汇排序后比较 max flow
sort -n -k1,1 -k2,2 "$R1" > "$TMP/r1.sorted"
sort -n -k1,1 -k2,2 "$R2" > "$TMP/r2.sorted"
if diff -q "$TMP/r1.sorted" "$TMP/r2.sorted" >/dev/null 2>&1; then
    echo "[PASS] 查询独立性：打乱顺序结果一致"
else
    echo "[FAIL] 查询独立性：结果不一致"
    echo "--- orig ---"; cat "$TMP/r1.sorted"
    echo "--- shuf ---"; cat "$TMP/r2.sorted"
    fail=1
fi

echo ""
if [ "$fail" -eq 0 ]; then
    echo "=========================================="
    echo "全部测试通过 ✓"
    echo "=========================================="
else
    echo "=========================================="
    echo "存在失败测试 ✗"
    echo "=========================================="
    exit 1
fi
