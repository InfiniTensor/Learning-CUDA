#!/usr/bin/env bash
# profile.sh —— ncu 与 nsys 分析。产出报告文件与可直接贴回的文本摘要。
#
# 本版力计算固定为 V3，所以只采样一个 kernel。
# 开发版 n_version5 的 profile.sh 会对 V0 与 V4 分别采样并排对比，
# 用来论证"tiling + 打包把算术强度从 1.25 flop/byte 提到 ~320，
# 于是从访存受限转入计算受限"；那份对比数据见 n_bodyversion.md §3.1–§3.7。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build"
OUT="${ROOT}/results/profile"
mkdir -p "${OUT}"

# profiling 用一个中等规模 + 少量步数：ncu 会把每个 kernel 重放多次，
# 步数多了会跑很久，而 kernel 的行为与步数无关。
N="${NBODY_PROFILE_N:-65536}"
BLOCK="${NBODY_PROFILE_BLOCK:-128}"
ic="${ROOT}/data/uniform_${N}.txt"
[ -f "${ic}" ] || "${BIN}/nbody_ic" --preset uniform -n "${N}" --out "${ic}"

cat > "${OUT}/params_profile.txt" <<EOF
dt = 1e-3
num_steps = 20
record_interval = 20
G = 1.0
softening = 0.01
integrator = "leapfrog"
EOF

# ncu 的关键指标。抽出来成变量，ncu 与摘要两处共用同一张清单，
# 避免"采了指标但摘要里没打印"这种低级不一致。
METRICS='DRAM Throughput|Compute \(SM\) Throughput|Memory Throughput|Achieved Occupancy|Duration|Registers Per Thread|SM Frequency|Block Size|Shared Memory'

echo "=== ncu: speed of light + roofline ==="
if command -v ncu > /dev/null 2>&1; then
  # 只 profile 力计算 kernel（-k regex），跳过 elementwise 的积分器 kernel。
  # --launch-count 3 而非全部：kernel 行为与步数无关，采 3 次足够，
  # 而 ncu 每次都要重放 kernel，采满 20 步会拖很久。
  ncu --set full \
      -k regex:KernelComputeAcc \
      --launch-count 3 \
      -f -o "${OUT}/force_v3" \
      "${BIN}/nbody_sim" \
        --bodies "${ic}" \
        --params "${OUT}/params_profile.txt" \
        --block "${BLOCK}" \
        --out "${OUT}/traj_v3.bin" \
      > "${OUT}/ncu_run.log" 2>&1
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "  ncu failed (exit ${rc}); see ${OUT}/ncu_run.log"
    # ERR_NVGPUCTRPERM = 云平台未放开性能计数器权限。这是本项目已经遇到过的
    # 情况，明确点出来省得再排查一遍。应对方式：换到有 profiling 权限的
    # 环境重跑本脚本（无需改代码），或让宿主机设置
    # NVreg_RestrictProfilingToAdminUsers=0。
    grep -qi 'ERR_NVGPUCTRPERM\|permission' "${OUT}/ncu_run.log" && \
      echo "  -> performance counters are blocked by the driver policy" && \
      echo "     (host needs NVreg_RestrictProfilingToAdminUsers=0)"
  else
    echo "  wrote ${OUT}/force_v3.ncu-rep"

    # 文本摘要：贴回时用这个，不用传二进制 rep。
    ncu --set full -k regex:KernelComputeAcc --launch-count 1 \
        --print-summary per-kernel \
        "${BIN}/nbody_sim" \
          --bodies "${ic}" \
          --params "${OUT}/params_profile.txt" \
          --block "${BLOCK}" \
          --out "${OUT}/traj_v3.bin" \
        > "${OUT}/ncu_summary.txt" 2>&1

    grep -E "${METRICS}" "${OUT}/ncu_summary.txt" | sed 's/^/    /' | head -20
  fi
  rm -f "${OUT}/traj_v3.bin"
else
  echo "  ncu not found, skipping"
fi

echo
echo "=== nsys: timeline, kernel mix, memcpy overlap ==="
if command -v nsys > /dev/null 2>&1; then
  nsys profile -t cuda,nvtx,osrt \
       -o "${OUT}/timeline_v3" -f true \
       "${BIN}/nbody_sim" \
         --bodies "${ic}" \
         --params "${OUT}/params_profile.txt" \
         --block "${BLOCK}" \
         --out "${OUT}/traj_v3.bin" \
       > "${OUT}/nsys_run.log" 2>&1
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "  nsys failed (exit ${rc}); see ${OUT}/nsys_run.log"
  else
    nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
         "${OUT}/timeline_v3.nsys-rep" > "${OUT}/nsys_stats.txt" 2>&1
    sed -n '1,30p' "${OUT}/nsys_stats.txt" | sed 's/^/    /'
    echo "  wrote ${OUT}/timeline_v3.nsys-rep"
  fi
  rm -f "${OUT}/traj_v3.bin"
else
  echo "  nsys not found, skipping"
fi

echo
echo "artifacts in ${OUT}/"
echo "  paste back: ncu_summary.txt + nsys_stats.txt"
echo "  keep on server: force_v3.ncu-rep, timeline_v3.nsys-rep (open in GUI if needed)"
