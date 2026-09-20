#!/bin/bash
# ============================================================
# tests/run_tests.sh —— 选题五 参数兼容性与健壮性测试
#
# 覆盖"参数输入"这一层最容易出问题的地方：题目文档给出的参数文件
# 用的是 option_type / risk_free_rate，取值还带引号且为小写枚举；
# 若解析器只认项目内部命名，会静默回落到结构体默认值，算出一个
# 看起来合理但完全错误的价格。本脚本把这类静默失败钉成回归测试。
#
# 用法: bash tests/run_tests.sh [SM_ARCH]
# ============================================================
exec 2>&1
set -u
cd "$(dirname "$0")/.." || exit 2

BIN=./build/cuda_pricing
TMP=tests/tmp
mkdir -p "$TMP"
fail=0

step() { echo ""; echo "------------------------------------------"; echo "$1"; echo "------------------------------------------"; }
ok()   { echo "[PASS] $1"; }
bad()  { echo "[FAIL] $1"; fail=1; }

# ---------- 0. 构建 ----------
step "[0/5] 构建"
bash build.sh >/dev/null 2>&1 && ok "构建成功" || { bad "构建失败"; exit 1; }

# ---------- 1. 题目文档格式的参数文件必须被正确解析 ----------
step "[1/5] 题目文档格式参数解析（option_type / risk_free_rate / 带引号小写枚举）"
cat > "$TMP/spec_opt.txt" <<'EOF'
option_type = barrier_call
spot = 100.0
strike = 100.0
risk_free_rate = 0.05
volatility = 0.25
maturity = 2.0
barrier = 120.0
barrier_dir = up
barrier_type = knock_out
EOF
cat > "$TMP/spec_sim.txt" <<'EOF'
num_paths = 200000
num_steps = 128
seed = 7
rng = "curand"
variance_reduction = "antithetic"
EOF

out=$("$BIN" "$TMP/spec_opt.txt" "$TMP/spec_sim.txt" "$TMP/spec.json" --json --no-cpu 2>&1)
echo "$out" | grep -E '^Option:|^Sim:' | sed 's/^/    /'
# r=0.05 / sigma=0.25 / T=2 与默认值(0.03/0.2/1)不同，可证明键名确实生效
echo "$out" | grep -q "BARRIER_CALL" && echo "$out" | grep -q "r=0.05" \
    && echo "$out" | grep -q "sigma=0.25" && echo "$out" | grep -q "T=2" \
    && ok "题目格式键名被正确解析（未回落到默认值）" \
    || bad "题目格式键名未被解析"
# VR=1 表示 ANTITHETIC（带引号 + 小写）
echo "$out" | grep -q "VR=1" && ok "带引号的小写枚举 variance_reduction=\"antithetic\" 被识别" \
                             || bad "小写/带引号枚举未被识别"

# ---------- 2. 未识别的键名必须告警 ----------
step "[2/5] 未识别键名告警"
printf 'option_type = european_call\nrisk_free_rate = 0.03\nbogus_key = 1\n' > "$TMP/warn_opt.txt"
out=$("$BIN" "$TMP/warn_opt.txt" "$TMP/spec_sim.txt" "$TMP/warn.json" --json --no-cpu 2>&1)
echo "$out" | grep -q "未识别的参数键" && ok "未识别键名有告警" || bad "未识别键名静默忽略"

# ---------- 3. 参数文件缺失必须报错（而不是用默认值继续跑） ----------
step "[3/5] 参数文件缺失"
out=$("$BIN" "$TMP/no_such_file.txt" "$TMP/spec_sim.txt" "$TMP/x.json" --json --no-cpu 2>&1)
rc=$?
if [ $rc -ne 0 ] && echo "$out" | grep -q "无法打开参数文件"; then
    ok "缺失文件明确报错并返回非零退出码 (rc=$rc)"
else
    bad "缺失文件未报错 (rc=$rc)，会用默认参数继续定价"
fi

# ---------- 4. 非法枚举值 / 矛盾障碍配置必须报错 ----------
step "[4/5] 非法取值与矛盾配置"
printf 'option_type = asian_arithmetic\nspot = 100\nstrike = 100\nrisk_free_rate = 0.03\nvolatility = 0.2\nmaturity = 1\n' > "$TMP/bad_type.txt"
out=$("$BIN" "$TMP/bad_type.txt" "$TMP/spec_sim.txt" "$TMP/x.json" --json --no-cpu 2>&1)
rc=$?
[ $rc -ne 0 ] && echo "$out" | grep -q "无法识别的 option_type" \
    && ok "非法 option_type 明确报错" || bad "非法 option_type 未报错"

sed 's/barrier_dir = up/barrier_dir = down/' "$TMP/spec_opt.txt" > "$TMP/bad_barrier.txt"
out=$("$BIN" "$TMP/bad_barrier.txt" "$TMP/spec_sim.txt" "$TMP/x.json" --json --no-cpu 2>&1)
rc=$?
[ $rc -ne 0 ] && echo "$out" | grep -q "校验失败" \
    && ok "矛盾障碍配置明确报错（避免静默输出 price=0/err=0 的假完美结果）" \
    || bad "矛盾障碍配置未报错"

# ---------- 5. 回归测试（期权定价正确性 + CPU/GPU 对比） ----------
step "[5/5] 定价正确性回归（6 类期权）"
if bash regression_test.sh 300000 >"$TMP/regression.log" 2>&1; then
    sed -n '/Summary Table/,/^$/p' "$TMP/regression.log" | sed 's/^/    /'
    ok "回归测试通过"
else
    bad "回归测试失败"
    tail -20 "$TMP/regression.log"
fi

echo ""
if [ $fail -eq 0 ]; then
    echo "=========================================="; echo "全部测试通过 ✓"; echo "=========================================="
else
    echo "=========================================="; echo "存在失败测试 ✗"; echo "=========================================="; exit 1
fi
