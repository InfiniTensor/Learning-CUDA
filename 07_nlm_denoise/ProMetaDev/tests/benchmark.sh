#!/bin/bash
# ============================================================
# tests/benchmark.sh —— NLM 性能与质量基准
#   ① 分辨率扫描（含题目要求的 1920x1080 与进阶目标 4K）
#   ② 滤波参数扫描（patch_radius / search_radius / h）
#   ③ 近似开关的质量-速度权衡
#   ④ 灰度/RGB 通道对比
#
# 图像与二进制放在 Linux 原生路径（/tmp）下运行：工作区在 /mnt/c，
# 走 Windows 挂载时 PNG 读写会慢一个数量级，会掩盖真实计算耗时。
#
# 每个配置重复 REP 次取**中位数**：本机是笔记本 GPU，同一配置单次采样
# 波动可达 ±50%（显存分配、时钟、调度），单次数字不能直接写进报告。
#
# 用法: bash tests/benchmark.sh [SM_ARCH]
#   REP=5 bash tests/benchmark.sh 90          # 自定义重复次数
#   ONLY=1 bash tests/benchmark.sh 90         # 只跑第①节（冷态测量时用，避免前置负载把 GPU 烤热）
# ============================================================
exec 2>&1
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SM_ARCH="${1:-90}"
REP="${REP:-3}"
ONLY="${ONLY:-1234}"
has() { [[ "$ONLY" == *"$1"* ]]; }
cd "$ROOT" || exit 2

echo "构建 (sm_${SM_ARCH}) ..."
bash build.sh "$SM_ARCH" >/dev/null 2>&1 || { echo "构建失败"; exit 1; }

WORK=/tmp/nlm_bench
mkdir -p "$WORK"
cp build/nlm "$WORK/"
cd "$WORK" || exit 2

gen() {  # gen <尺寸> <通道> <sigma> <seed> <前缀>
    [ -f "$5_clean.png" ] && return
    python3 "$ROOT/tools/gen_image.py" --size "$1" --channels "$2" --sigma "$3" --seed "$4" \
        --out-clean "$5_clean.png" --out-noisy "$5_noisy.png" >/dev/null
}

# 从程序输出中抽取指标
pick() { echo "$1" | grep -F "$2" | head -1 | sed -n "s/.*$2 *\([0-9.]*\).*/\1/p"; }
psnr_of() { echo "$1" | grep -F "vs 无噪真值" | head -1 |
            sed -n 's/.*PSNR= *\([0-9.]*\).*/\1/p'; }
med() { sort -n | awk '{a[NR]=$1} END{ if(NR==0){print "nan"} else if(NR%2){printf "%.2f",a[(NR+1)/2]} else {printf "%.2f",(a[NR/2]+a[NR/2+1])/2} }'; }

# 重复 REP 次，回显 "<合计中位数> <核函数中位数> <吞吐MPix/s中位数> <PSNR(最后一次)>"
run_rep() {   # run_rep <in.png> <params.txt> <out.png> [额外选项...]
    local tot="" ker="" thp="" out=""
    for _ in $(seq 1 "$REP"); do
        out=$(./nlm denoise "$1" "$2" "$3" "${@:4}")
        tot="$tot$(pick "$out" "合计")\n"
        ker="$ker$(pick "$out" "核函数")\n"
        thp="$thp$(echo "$out" | sed -n 's/.*吞吐 *\([0-9.]*\) MPix.*/\1/p')\n"
    done
    printf '%s %s %s %s' \
        "$(printf "$tot" | med)" "$(printf "$ker" | med)" "$(printf "$thp" | med)" "$(psnr_of "$out")"
}

# 记录 GPU 时钟/温度：笔记本 GPU 在持续负载下会热降频，同一负载冷热状态可差 2 倍以上，
# 不记录时钟的话性能数字无法解释、也无法复现。
clock_line() {
    nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw \
        --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /'
}

echo ""
echo "重复次数 REP=$REP（取中位数）；GPU 型号与时钟随每次测试记录。"
echo "初始 GPU 状态:"; clock_line

# ============================================================
echo ""
if has 1; then
echo "===== ① 分辨率扫描（patch_radius=3, search_radius=10, h=10, sigma=25, RGB）====="
printf "%-16s %10s %10s %10s %12s %12s %10s\n" "分辨率" "GPU总ms" "内核ms" "MPix/s" "PSNR_vs真值" "OpenCV ms" "加速比"
for spec in "480x270:tiny" "1280x720:hd" "1920x1080:fhd" "3840x2160:uhd"; do
    size="${spec%%:*}"; tag="${spec##*:}"
    gen "$size" 3 25 5 "$tag"
    printf 'patch_radius = 3\nsearch_radius = 10\nh = 10.0\nsigma = 25.0\n' > p.txt
    read -r gpu ker thp psnr <<<"$(run_rep "${tag}_noisy.png" p.txt "${tag}_out.png" \
        --opencv --ref "${tag}_clean.png")"
    # OpenCV 只跑一次（纯 CPU、且相对稳定），取它的耗时与加速比
    out=$(./nlm denoise "${tag}_noisy.png" p.txt "${tag}_out.png" --opencv --ref "${tag}_clean.png")
    ocv=$(echo "$out" | sed -n 's/.*\[opencv\] 参考耗时 *\([0-9.]*\) ms.*/\1/p')
    sp=$(awk -v a="$ocv" -v b="$gpu" 'BEGIN{printf "%.1f", a/b}')
    printf "%-16s %10s %10s %10s %12s %12s %10s\n" "$size" "$gpu" "$ker" "$thp" "$psnr" "$ocv" "$sp"
done
fi

# ============================================================
echo ""
echo "===== ② 滤波参数扫描（1920x1080 RGB, sigma=25）====="
printf "%-8s %-8s %-7s %10s %12s %12s\n" "rp" "rs" "h" "GPU ms" "MPix/s" "PSNR_vs真值"
if has 2; then
for cfg in "1 10 10" "2 10 10" "3 10 10" "5 10 10" "3 5 10" "3 15 10" "3 10 5" "3 10 20"; do
    set -- $cfg
    printf 'patch_radius = %s\nsearch_radius = %s\nh = %s\nsigma = 25.0\n' "$1" "$2" "$3" > p.txt
    read -r gpu ker thp psnr <<<"$(run_rep fhd_noisy.png p.txt fhd_cfg.png --ref fhd_clean.png)"
    printf "%-8s %-8s %-7s %10s %12s %12s\n" "$1" "$2" "$3" "$gpu" "$thp" "$psnr"
done
fi

# ============================================================
echo ""
echo "===== ③ 近似开关的质量-速度权衡（1920x1080 RGB, rp=3, rs=10, h=10, sigma=25）====="
printf "%-24s %10s %12s %12s\n" "模式" "GPU ms" "MPix/s" "PSNR_vs真值"
if has 3; then
run_mode() {  # run_mode <名称> <额外参数>
    { printf 'patch_radius = 3\nsearch_radius = 10\nh = 10.0\nsigma = 25.0\n'; [ -n "$2" ] && printf '%s\n' "$2"; } > p.txt
    local gpu ker thp psnr
    read -r gpu ker thp psnr <<<"$(run_rep fhd_noisy.png p.txt fhd_mode.png --ref fhd_clean.png)"
    printf "%-24s %10s %12s %12s\n" "$1" "$gpu" "$thp" "$psnr"
}
run_mode "精确 (exact)"        ""
run_mode "search_step=2"       "search_step = 2"
run_mode "search_step=3"       "search_step = 3"
run_mode "patch_step=2"        "patch_step = 2"
run_mode "LUT(2048)"           "use_lut = true
lut_bins = 2048"
run_mode "search_step=2 + LUT" "search_step = 2
use_lut = true
lut_bins = 2048"
fi

# ============================================================
echo ""
echo "===== ④ 灰度 vs RGB（1920x1080, rp=3, rs=10, h=10, sigma=25）====="
printf "%-10s %10s %12s %12s %14s\n" "通道" "GPU ms" "MPix/s" "PSNR" "OpenCV ms"
if has 4; then
for ch in 1 3; do
    tag="cmp_c${ch}"
    gen 1920x1080 "$ch" 25 6 "$tag"
    printf 'patch_radius = 3\nsearch_radius = 10\nh = 10.0\nsigma = 25.0\n' > p.txt
    read -r gpu ker thp psnr <<<"$(run_rep "${tag}_noisy.png" p.txt "${tag}_out.png" \
        --opencv --ref "${tag}_clean.png")"
    out=$(./nlm denoise "${tag}_noisy.png" p.txt "${tag}_out.png" --opencv --ref "${tag}_clean.png")
    ocv=$(echo "$out" | sed -n 's/.*\[opencv\] 参考耗时 *\([0-9.]*\) ms.*/\1/p')
    label=$([ "$ch" = 1 ] && echo "灰度(1ch)" || echo "彩色(3ch)")
    printf "%-10s %10s %12s %12s %14s\n" "$label" "$gpu" "$thp" "$psnr" "$ocv"
done
fi

echo ""
echo "硬件: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "终态 GPU 状态:"; clock_line
echo "CPU 核数: $(nproc)"
