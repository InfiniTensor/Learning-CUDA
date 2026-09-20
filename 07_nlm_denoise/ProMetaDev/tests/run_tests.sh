#!/bin/bash
# ============================================================
# tests/run_tests.sh —— NLM 降噪 端到端测试
#   1) 引擎自检
#   2) GPU 与 CPU 参考实现逐像素一致性（灰度和彩色）
#   3) 降噪有效性（相对无噪真值的 PSNR/MAE 必须改善）
#   4) 近似开关（search_step / patch_step / LUT）的误差量化
#   5) 参数校验（非法输入必须报错）
# 用法: bash tests/run_tests.sh [SM_ARCH]
# ============================================================
exec 2>&1
set -u
cd "$(dirname "$0")/.." || exit 2

SM_ARCH="${1:-90}"
BIN=./build/nlm
GEN=tools/gen_image.py
TMP=tests/tmp
mkdir -p "$TMP"
fail=0

step() { echo ""; echo "------------------------------------------"; echo "$1"; echo "------------------------------------------"; }
ok()   { echo "[PASS] $1"; }
bad()  { echo "[FAIL] $1"; fail=1; }

# 从程序输出里取一个指标值： get <输出> <关键字>
metric() { echo "$1" | grep -F "$2" | head -1 | sed -n "s/.*$2[= ]*\([0-9.]*\).*/\1/p"; }

# ---------- 0. 构建 ----------
step "[0/6] 构建"
if bash build.sh "$SM_ARCH" >/dev/null 2>&1; then ok "构建成功"; else bad "构建失败"; exit 1; fi

# ---------- 1. 自检 ----------
step "[1/6] 引擎自检"
if "$BIN" selftest; then ok "selftest"; else bad "selftest"; fi

# ---------- 2. GPU vs CPU 参考（灰度 + 彩色） ----------
step "[2/6] GPU 与 CPU 参考实现一致性"
run_pair() {
    local name="$1" size="$2" ch="$3" sigma="$4" h="$5" rs="$6"
    python3 "$GEN" --size "$size" --channels "$ch" --sigma "$sigma" --seed 21 \
        --out-clean "$TMP/${name}_c.png" --out-noisy "$TMP/${name}_n.png" >/dev/null
    printf 'patch_radius = 3\nsearch_radius = %s\nh = %s\nsigma = %s\n' "$rs" "$h" "$sigma" \
        > "$TMP/${name}_p.txt"
    local out
    out=$("$BIN" denoise "$TMP/${name}_n.png" "$TMP/${name}_p.txt" "$TMP/${name}_o.png" \
            --cpu --ref "$TMP/${name}_c.png")
    local mae
    mae=$(metric "$out" "vs CPU 参考  MAE")
    echo "    $name: GPU vs CPU MAE=$mae"
    # 允许极小的浮点舍入差异：MAE 应 < 0.05
    if [ -n "$mae" ] && python3 -c "import sys; sys.exit(0 if float('$mae') < 0.05 else 1)"; then
        ok "$name GPU 与 CPU 参考一致（MAE=$mae）"
    else
        bad "$name GPU 与 CPU 参考不一致（MAE=$mae）"
    fi
}
run_pair "gray256"  "256x256" 1 20 12.0 8
run_pair "rgb384"   "384x384" 3 20 12.0 8

# ---------- 3. 降噪有效性 ----------
step "[3/6] 降噪有效性（相对无噪真值）"
python3 "$GEN" --size 384x384 --channels 3 --sigma 25 --seed 33 \
    --out-clean "$TMP/eff_c.png" --out-noisy "$TMP/eff_n.png" >/dev/null
printf 'patch_radius = 3\nsearch_radius = 10\nh = 10.0\nsigma = 25.0\n' > "$TMP/eff_p.txt"
out=$("$BIN" denoise "$TMP/eff_n.png" "$TMP/eff_p.txt" "$TMP/eff_o.png" \
        --opencv --ref "$TMP/eff_c.png")
echo "$out" | grep -E 'vs 无噪真值|OpenCV vs 真值' | sed 's/^/    /'
psnr_gpu=$(metric "$out" "vs 无噪真值 MAE" 2>/dev/null)
psnr_out=$(echo "$out" | grep -F "vs 无噪真值" | sed -n 's/.*PSNR= *\([0-9.]*\).*/\1/p')
psnr_noisy=$(python3 -c "
import numpy as np
from PIL import Image
c = np.asarray(Image.open('$TMP/eff_c.png'), np.float64)
n = np.asarray(Image.open('$TMP/eff_n.png'), np.float64)
mse = ((c-n)**2).mean()
print(f'{10*np.log10(255*255/mse):.2f}' if mse > 0 else '99')
")
echo "    含噪输入 PSNR=$psnr_noisy dB  →  降噪后 PSNR=$psnr_out dB"
if python3 -c "
import sys
sys.exit(0 if float('$psnr_out') > float('$psnr_noisy') + 8.0 else 1)"; then
    ok "PSNR 相比含噪输入提升超过 8 dB"
else
    bad "PSNR 提升不足（$psnr_noisy → $psnr_out）"
fi

# ---------- 4. 近似开关的误差量化 ----------
step "[4/6] 近似开关的误差与加速"
printf 'patch_radius = 3\nsearch_radius = 10\nh = 10.0\nsigma = 25.0\n' > "$TMP/ex_p.txt"
printf 'patch_radius = 3\nsearch_radius = 10\nh = 10.0\nsigma = 25.0\nsearch_step = 2\n' > "$TMP/ap_s_p.txt"
printf 'patch_radius = 3\nsearch_radius = 10\nh = 10.0\nsigma = 25.0\npatch_step = 2\n' > "$TMP/ap_p_p.txt"
printf 'patch_radius = 3\nsearch_radius = 10\nh = 10.0\nsigma = 25.0\nuse_lut = true\nlut_bins = 2048\n' > "$TMP/ap_l_p.txt"

python3 - "$BIN" "$TMP" <<'PY' > "$TMP/ap_report.txt"
import subprocess, sys
import numpy as np
from PIL import Image
bin_, tmp = sys.argv[1], sys.argv[2]

def run(name, pfile, ofile):
    out = subprocess.run([bin_, 'denoise', f'{tmp}/eff_n.png', pfile, ofile],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if '合计' in line:
            return float(line.split('合计')[1].split('ms')[0].strip())
    return float('nan')

def psnr(a, b):
    x = np.asarray(Image.open(a), np.float64); y = np.asarray(Image.open(b), np.float64)
    d = ((x - y) ** 2).mean()
    return 99.0 if d < 1e-12 else float(10 * np.log10(255 * 255 / d))

def mae(a, b):
    return float(np.abs(np.asarray(Image.open(a), np.float64) -
                        np.asarray(Image.open(b), np.float64)).mean())

clean = f'{tmp}/eff_c.png'
ex_img = f'{tmp}/ap_exact.png'
ex_ms = run('exact', f'{tmp}/ex_p.txt', ex_img)
rows = [('exact', ex_img, ex_ms)]
for name, pfile, ofile in [
        ('search_step2', f'{tmp}/ap_s_p.txt', f'{tmp}/ap_s2.png'),
        ('patch_step2',  f'{tmp}/ap_p_p.txt', f'{tmp}/ap_p2.png'),
        ('lut2048', f'{tmp}/ap_l_p.txt', f'{tmp}/ap_lut.png')]:
    ms = run(name, pfile, ofile)
    rows.append((name, ofile, ms))

print(f"{'mode':<16}{'GPU_ms':>9}{'MAE_vs_exact':>14}{'PSNR_vs_truth':>15}")
for name, img, ms in rows:
    print(f"{name:<16}{ms:>9.2f}{mae(img, ex_img):>14.4f}{psnr(img, clean):>15.2f}")
# 供 shell 解析的机器可读行（逗号分隔，名字无空格）
for name, img, ms in rows:
    print(f"CSV,{name},{ms:.2f},{mae(img, ex_img):.4f},{psnr(img, clean):.2f}")
PY
sed 's/^/    /' <(grep -v '^CSV,' "$TMP/ap_report.txt")

val() { grep "^CSV,$1," "$TMP/ap_report.txt" | cut -d, -f$2; }
exact_ms=$(val exact 3); s2_ms=$(val search_step2 3)
exact_psnr=$(val exact 5); s2_psnr=$(val search_step2 5)
p2_psnr=$(val patch_step2 5); lut_psnr=$(val lut2048 5)
s2_mae=$(val search_step2 4); p2_mae=$(val patch_step2 4); lut_mae=$(val lut2048 4)

# 判据：① LUT 属于"近无损"加速，MAE_vs_exact 必须 < 0.5；
#       ② search_step / patch_step 属于**有损**的质量-速度旋钮，允许改变结果，
#          但必须仍然"有效降噪"（相对无噪真值的 PSNR 高于含噪输入 8 dB 以上），
#          并如实打印相对精确模式的质量损失。
near_lossless() {
    if [ -n "$2" ] && python3 -c "import sys; sys.exit(0 if float('$2') < 0.5 else 1)" 2>/dev/null; then
        ok "$1 近乎无损（MAE_vs_exact=$2，PSNR $exact_psnr → $3 dB）"
    else
        bad "$1 误差过大（MAE_vs_exact=$2）"
    fi
}
still_denoises() {  # still_denoises <名称> <该模式PSNR> <mae_vs_exact>
    local loss
    loss=$(awk -v a="$exact_psnr" -v b="$2" 'BEGIN{printf "%.2f", a-b}')
    if python3 -c "import sys; sys.exit(0 if float('$2') >= float('$psnr_noisy') + 8.0 else 1)" 2>/dev/null; then
        ok "$1 仍有效降噪（PSNR $psnr_noisy(含噪) → $2 dB；相对精确模式损失 ${loss} dB）"
    else
        bad "$1 降噪失效（PSNR 仅 $2 dB）"
    fi
}
near_lossless "LUT(2048)"     "$lut_mae" "$lut_psnr"
still_denoises "search_step=2" "$s2_psnr" "$s2_mae"
still_denoises "patch_step=2"  "$p2_psnr" "$p2_mae"

# 近似模式应当更快（允许小幅波动，只要不是变慢）
if [ -n "$exact_ms" ] && [ -n "$s2_ms" ] && \
   python3 -c "import sys; sys.exit(0 if float('$s2_ms') <= float('$exact_ms')*1.05 else 1)"; then
    ok "search_step=2 未变慢（精确 ${exact_ms}ms → ${s2_ms}ms）"
else
    bad "search_step=2 反而更慢（${exact_ms}ms → ${s2_ms}ms）"
fi

# ---------- 5. 参数校验 ----------
step "[5/6] 参数校验"
printf 'patch_radius = 0\nsearch_radius = 10\nh = 10\nsigma = 25\n' > "$TMP/bad1.txt"
"$BIN" denoise "$TMP/eff_n.png" "$TMP/bad1.txt" "$TMP/x.png" >/dev/null 2>&1 \
    && bad "patch_radius=0 未报错" || ok "patch_radius=0 已报错"
printf 'patch_radius = 3\nsearch_radius = 10\nh = -1\nsigma = 25\n' > "$TMP/bad2.txt"
"$BIN" denoise "$TMP/eff_n.png" "$TMP/bad2.txt" "$TMP/x.png" >/dev/null 2>&1 \
    && bad "h=-1 未报错" || ok "h=-1 已报错"
"$BIN" denoise "$TMP/not_exist.png" "$TMP/eff_p.txt" "$TMP/x.png" >/dev/null 2>&1 \
    && bad "输入图不存在未报错" || ok "输入图不存在已报错"

# ---------- 6. 结果确定性 ----------
step "[6/6] 结果确定性（重复运行逐字节一致）"
"$BIN" denoise "$TMP/eff_n.png" "$TMP/eff_p.txt" "$TMP/rep1.png" >/dev/null
"$BIN" denoise "$TMP/eff_n.png" "$TMP/eff_p.txt" "$TMP/rep2.png" >/dev/null
if cmp -s "$TMP/rep1.png" "$TMP/rep2.png"; then ok "重复运行输出逐字节一致"; else bad "重复运行输出不一致"; fi

echo ""
if [ "$fail" -eq 0 ]; then
    echo "=========================================="; echo "全部测试通过 ✓"; echo "=========================================="
else
    echo "=========================================="; echo "存在失败测试 ✗"; echo "=========================================="; exit 1
fi
