#!/usr/bin/env bash
# 端到端测试：精确检索正确性（K 覆盖）、IVF 索引合法性、nprobe/batch 扫描、
# 三种度量（L2 / 内积 / cosine）
set -uo pipefail
cd "$(dirname "$0")/.."

BIN="${BIN:-./build/vs}"
OUT="${OUT:-outputs/tests}"
N="${N:-20000}"; DIM="${DIM:-128}"; NQ="${NQ:-200}"; NLIST="${NLIST:-64}"
mkdir -p "$OUT/data" "$OUT/results"
[ -x "$BIN" ] || { echo "未找到 $BIN，请先执行 bash build.sh"; exit 2; }

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); echo "  [PASS] $1"; }
bad() { FAIL=$((FAIL+1)); echo "  [FAIL] $1"; }

echo "=== 0. 引擎自检 ==="
if "$BIN" selftest > /tmp/vs_st.log 2>&1; then ok "selftest"; else bad "selftest"; tail -6 /tmp/vs_st.log | sed 's/^/      /'; fi

echo
echo "=== 1. 生成数据 (clustered, n=$N dim=$DIM nq=$NQ nlist=$NLIST) ==="
BASE="$OUT/data/base.txt"; Q="$OUT/data/q.txt"
"$BIN" gen clustered "$N" "$DIM" "$BASE" --nlist "$NLIST" --seed 7 > /dev/null \
    && ok "生成向量库" || bad "生成向量库"
"$BIN" genq "$BASE" "$NQ" "$Q" --kind clustered --seed 99 > /dev/null \
    && ok "生成查询集" || bad "生成查询集"

echo
echo "=== 2. 精确检索：K ∈ {1,10,50,100}，GPU vs CPU 参考 ==="
for K in 1 10 50 100; do
    C=/tmp/vs_k$K.cfg
    printf 'top_k = %d\nsearch_mode = "exact"\nbatch_size = 64\n' "$K" > "$C"
    if "$BIN" exact "$BASE" "$Q" "$C" "$OUT/results/exact_k$K" > /tmp/vs_ex.log 2>&1; then
        ok "exact K=$K 与 CPU 参考逐名次一致"
    else
        bad "exact K=$K"
        grep -E '不一致|max_diff' /tmp/vs_ex.log | head -3 | sed 's/^/      /'
    fi
done

echo
echo "=== 3. 建 IVF 索引 + 倒排表合法性 ==="
IVF=/tmp/vs_ivf.cfg
cat > "$IVF" <<EOF
top_k = 10
search_mode = "ivf_flat"
batch_size = 64
nlist = $NLIST
nprobe = 8
kmeans_iters = 10
seed = 1234
EOF
if "$BIN" build "$BASE" "$IVF" "$OUT/results/index.bin" > /tmp/vs_build.log 2>&1; then
    if grep -q '重复=0, 缺失=0' /tmp/vs_build.log; then
        ok "倒排表是 0..n-1 的合法排列"
    else
        bad "倒排表校验失败"; grep '校验' /tmp/vs_build.log | sed 's/^/      /'
    fi
else
    bad "建索引失败"
fi

echo
echo "=== 4. nprobe 扫描：召回率应随 nprobe 上升，nprobe=nlist 时达 1.0 ==="
"$BIN" bench "$BASE" "$Q" "$OUT/results/index.bin" "$IVF" "$OUT/results/bench" \
    --gt "$OUT/results/exact_k10.result" > /tmp/vs_bench.log 2>&1
# 只取 nprobe 段（形如 "nprobe recall qps ..."）
sed -n '/\[nprobe 扫描\]/,/\[batch_size 扫描\]/p' /tmp/vs_bench.log \
    | awk '/^[0-9]+[ \t]+[0-9.]+/{print $1, $2}' > /tmp/vs_nprobe.txt
if awk -v nl="$NLIST" '$1==nl && $2+0 >= 0.99999 {found=1} END{exit !found}' /tmp/vs_nprobe.txt; then
    ok "nprobe=nlist 时 recall@10 = 1.0"
else
    bad "nprobe=nlist 未达 1.0"
fi
if awk 'NR>1 && $2+0 < prev-1e-9 {bad=1} {prev=$2+0} END{exit bad?1:0}' /tmp/vs_nprobe.txt; then
    ok "召回率随 nprobe 单调不降"
else
    bad "召回率非单调"
fi

echo
echo "  扫描结果 (nprobe, recall@10):"
sed 's/^/      /' /tmp/vs_nprobe.txt

echo
echo "=== 5. 三种度量：L2 / 内积 / cosine（精确检索 GPU vs CPU）==="
for M in l2 inner_product cosine; do
    B2="$OUT/data/base_$M.txt"; Q2="$OUT/data/q_$M.txt"
    "$BIN" gen clustered "$N" "$DIM" "$B2" --nlist "$NLIST" --seed 7 --metric "$M" > /dev/null 2>&1
    "$BIN" genq "$B2" "$NQ" "$Q2" --kind clustered --seed 99 > /dev/null 2>&1
    if "$BIN" exact "$B2" "$Q2" /tmp/vs_k10.cfg "$OUT/results/exact_$M" > /tmp/vs_m.log 2>&1; then
        ok "度量 $M：精确检索与 CPU 一致"
    else
        bad "度量 $M"
        grep -E '不一致' /tmp/vs_m.log | head -2 | sed 's/^/      /'
    fi
done

echo
echo "=== 汇总 ==="
echo "PASS=$PASS  FAIL=$FAIL"
[ $FAIL -eq 0 ] && echo "全部通过" || echo "存在失败用例"
exit $([ $FAIL -eq 0 ] && echo 0 || echo 1)
