#!/usr/bin/env bash
# remote_all.sh —— 远程侧唯一入口。跑完全部流程并打印一份可直接贴回的摘要。
#
#   bash scripts/remote_all.sh              # 全流程
#   bash scripts/remote_all.sh quick        # 只 build + validate（省机时）
#
# 设计意图：租的机器按小时计费，每次往返成本高，
# 所以一条命令产出全部数据 + 一份紧凑摘要，不需要交互。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-full}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${ROOT}/results/remote_all_${STAMP}.log"
mkdir -p "${ROOT}/results"
mkdir -p "${ROOT}/video"

# 全部输出既进屏幕也进日志，方便贴回。
exec > >(tee "${LOG}") 2>&1

echo "########################################"
echo "# nbody remote run: ${STAMP} (mode=${MODE})"
echo "########################################"
echo

build_rc=1
validate_rc=1
bench_rc=1

echo "########## BUILD ##########"
bash "${ROOT}/scripts/build.sh"
build_rc=$?
if [ ${build_rc} -ne 0 ]; then
  echo
  echo "BUILD FAILED — stopping here. Paste the errors above back."
  exit 1
fi

echo
echo "########## VALIDATE ##########"
bash "${ROOT}/scripts/validate.sh"
validate_rc=$?

vis_rc=0
energy_rc=0
scale_rc=0

if [ "${MODE}" = "quick" ]; then
  echo
  echo "quick mode: skipping bench, profile, and visualizations"
else
  echo
  echo "########## BENCH ##########"
  bash "${ROOT}/scripts/bench.sh"
  bench_rc=$?

  echo
  echo "########## PROFILE ##########"
  bash "${ROOT}/scripts/profile.sh"

  # -------------------------------------------------------------------------
  # 三种现象：模拟 + 动画
  # -------------------------------------------------------------------------
  echo
  echo "########## PHENOMENA: cluster (N=65536) ##########"
  "${ROOT}/build/nbody_ic" --preset plummer -n 65536 --seed 20260821 \
      --out "${ROOT}/results/plummer_65536.txt"
  "${ROOT}/build/nbody_sim" --bodies "${ROOT}/results/plummer_65536.txt" \
      --params "${ROOT}/data/params_cluster.txt" \
      --block 128 --out "${ROOT}/results/cluster.bin" --log "${ROOT}/results/cluster.json"
  python3 "${ROOT}/tools/visualize.py" "${ROOT}/results/cluster.bin" \
      --dim 3 --trail 30 --save "${ROOT}/video/cluster.mp4" || vis_rc=$?

  echo
  echo "########## PHENOMENA: two-body (N=2) ##########"
  "${ROOT}/build/nbody_ic" --preset two_body --out "${ROOT}/results/two_body.txt"
  "${ROOT}/build/nbody_sim" --bodies "${ROOT}/results/two_body.txt" \
      --params "${ROOT}/data/params_two_body.txt" \
      --block 128 --out "${ROOT}/results/two_body.bin" --log "${ROOT}/results/two_body.json"
  python3 "${ROOT}/tools/visualize.py" "${ROOT}/results/two_body.bin" \
      --dim 2 --trail 60 --save "${ROOT}/video/two_body.mp4" || vis_rc=$?

  # Euler 对照
  "${ROOT}/build/nbody_sim" --bodies "${ROOT}/results/two_body.txt" \
      --params "${ROOT}/data/params_two_body_euler.txt" \
      --block 128 --out "${ROOT}/results/two_body_euler.bin" --log "${ROOT}/results/two_body_euler.json"
  python3 "${ROOT}/tools/visualize.py" "${ROOT}/results/two_body_euler.bin" \
      --dim 2 --trail 60 --save "${ROOT}/video/two_body_euler.mp4" || vis_rc=$?

  echo
  echo "########## PHENOMENA: collision (N=4096) ##########"
  "${ROOT}/build/nbody_ic" --preset cluster_collision -n 4096 --seed 20260821 \
      --sep 8 --vfrac 0.7 --impact-param 1.5 \
      --out "${ROOT}/results/collision_4096.txt"
  "${ROOT}/build/nbody_sim" --bodies "${ROOT}/results/collision_4096.txt" \
      --params "${ROOT}/data/params_collision.txt" \
      --block 128 --out "${ROOT}/results/collision.bin" --log "${ROOT}/results/collision.json"
  python3 "${ROOT}/tools/visualize.py" "${ROOT}/results/collision.bin" \
      --dim 3 --trail 40 --save "${ROOT}/video/collision.mp4" || vis_rc=$?

  # 能量对比图（用 validate 产出的日志）
  echo
  echo "########## CHARTS ##########"
  if [ -f "${ROOT}/results/validate/two_body_leapfrog.log" ] && \
     [ -f "${ROOT}/results/validate/two_body_euler.log" ]; then
    python3 "${ROOT}/tools/plot_energy.py" \
        "${ROOT}/results/validate/two_body_leapfrog.log" \
        "${ROOT}/results/validate/two_body_euler.log" \
        --out "${ROOT}/video/energy_comparison.png" || energy_rc=$?
  fi

  # 性能曲线图
  if [ -f "${ROOT}/results/bench/results.csv" ]; then
    python3 "${ROOT}/tools/plot_scaling.py" \
        --csv "${ROOT}/results/bench/results.csv" \
        --speedup "${ROOT}/results/bench/speedup_n65536.json" \
        --peak-tflops 106.6 --out "${ROOT}/video/scaling.png" || scale_rc=$?
  fi
fi

# ---------------------------------------------------------------------------
# 摘要：这一段是贴回给本地的核心内容。
# ---------------------------------------------------------------------------
echo
echo "########################################"
echo "# SUMMARY (paste this back)"
echo "########################################"
echo
echo "-- environment --"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
  --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
echo "cpu   : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
echo "cores : $(nproc)"
echo "nvcc  : $(nvcc --version | grep release | sed 's/.*release //')"
echo "gcc   : $(gcc --version | head -1 | awk '{print $NF}')"

echo
echo "-- status --"
printf 'build       : %s\n' "$([ ${build_rc} -eq 0 ] && echo OK || echo FAILED)"
printf 'validate    : %s\n' "$([ ${validate_rc} -eq 0 ] && echo OK || echo 'HAS FAILURES')"
if [ "${MODE}" != "quick" ]; then
  printf 'bench       : %s\n' "$([ ${bench_rc} -eq 0 ] && echo OK || echo FAILED)"
  printf 'visualize   : %s\n' "$([ ${vis_rc} -eq 0 ] && echo OK || echo FAILED)"
  printf 'energy plot : %s\n' "$([ ${energy_rc} -eq 0 ] && echo OK || echo FAILED)"
  printf 'scaling plot: %s\n' "$([ ${scale_rc} -eq 0 ] && echo OK || echo FAILED)"
fi

echo
echo "-- validation results --"
grep -E "^  .{42} (PASS|FAIL)" "${LOG}" 2>/dev/null | sed 's/^/  /' || true

echo
echo "-- two-body orbit --"
grep -hE "^two-body:" "${ROOT}"/results/validate/*.log 2>/dev/null | sed 's/^/  /' || true

echo
echo "-- energy behaviour --"
for f in "${ROOT}"/results/validate/two_body_leapfrog.log \
         "${ROOT}"/results/validate/two_body_euler.log \
         "${ROOT}"/results/validate/plummer_4096.log; do
  [ -f "$f" ] || continue
  printf '  %-24s %s\n' "$(basename "$f" .log)" \
    "$(grep -oE '\|dE/E0\|=[0-9.e+-]+' "$f" | head -1)"
done

if [ "${MODE}" != "quick" ] && [ -f "${ROOT}/results/bench/results.csv" ]; then
  echo
  echo "-- performance sweep --"
  cat "${ROOT}/results/bench/results.csv"

  echo
  echo "-- best config per N --"
  python3 - "${ROOT}/results/bench/results.csv" <<'PY'
import csv, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
best = {}
for r in rows:
    n = int(r["n"])
    ms = float(r["ms_per_step"])
    if n not in best or ms < float(best[n]["ms_per_step"]):
        best[n] = r
print(f'{"N":>8} {"block":>6} {"ms/step":>10} {"GFLOP/s":>10} {"p-steps/s":>12}')
for n in sorted(best):
    r = best[n]
    print(f'{n:>8} {r["block"]:>6} {float(r["ms_per_step"]):>10.4f} '
          f'{float(r["gflops"]):>10.1f} {float(r["particle_steps_per_sec"]):>12.3e}')
PY

  echo
  echo "-- cpu speedup (N=65536) --"
  grep -E "cpu_naive|cpu_scalar|cpu_openmp|OpenMP threads" \
    "${ROOT}/results/bench/speedup_n65536.log" 2>/dev/null | sed 's/^/  /' || true

  echo
  echo "-- ncu key metrics --"
  grep -E "DRAM Throughput|Compute \(SM\) Throughput|Achieved Occupancy|Duration|Registers Per Thread" \
    "${ROOT}/results/profile/ncu_summary.txt" 2>/dev/null | sed 's/^/  /' | head -12 || true
fi

echo
echo "########################################"
echo "full log: ${LOG}"
echo "artifacts: ${ROOT}/results/  (logs, csv, json)"
echo "videos   : ${ROOT}/video/    (mp4, png)"
echo "########################################"
