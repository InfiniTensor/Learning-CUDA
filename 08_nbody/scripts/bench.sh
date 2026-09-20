#!/usr/bin/env bash
# bench.sh —— 性能扫参。产出 results.csv 与逐项 json。
#
# 本版力计算固定为 V3，所以扫的是 N × block_size 二维网格。
# 开发版 n_version5 里还有第三个维度（kernel V0..V4 与 coarsen 1/2/4/8），
# 那份逐档对比数据见 n_bodyversion.md §3.1。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build"
OUT="${ROOT}/results/bench"
mkdir -p "${OUT}"

# N 的取值。4096 是作业的基础要求，65536 是进阶要求，262144 是我们的上限目标。
# 注意 4096 这一档测出的是 kernel launch 开销而非算力（单步仅 ~0.08 ms），
# 报告里要单独解释，不能和大 N 的数字混着看。
NS="${NBODY_BENCH_NS:-4096 16384 32768 65536 131072 262144}"
BLOCKS="${NBODY_BENCH_BLOCKS:-64 128 256 512}"

CSV="${OUT}/results.csv"
echo "n,block,kernel,precision,ms_per_step,total_ms,particle_steps_per_sec,gflops,mem_mib,smem_bytes,rel_energy_err,rel_angular_momentum_err" > "${CSV}"

for n in ${NS}; do
  ic="${ROOT}/data/uniform_${n}.txt"
  if [ ! -f "${ic}" ]; then
    echo "--- generating IC for N=${n} ---"
    "${BIN}/nbody_ic" --preset uniform -n "${n}" --out "${ic}"
  fi

  for block in ${BLOCKS}; do
    tag="n${n}_b${block}"
    echo "=== N=${n} block=${block} ==="
    # --out /dev/null：扫参不需要轨迹文件，直接丢弃省磁盘与写盘时间
    # （N=262144 时每个 315 MB，全组合会写满几十 GB）。
    "${BIN}/nbody_sim" \
      --bodies "${ic}" \
      --params "${ROOT}/data/params_bench.txt" \
      --block "${block}" \
      --out /dev/null \
      --log "${OUT}/${tag}.json" > "${OUT}/${tag}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      # block 太大导致 shared memory 超限是**预期**的失败
      # （smem 需求是 block*16B，FP64 下翻倍），程序会给出明确信息。
      # 这里不当成致命错误，继续扫下一个组合。
      echo "  SKIP/FAIL (exit ${rc}): $(tail -2 "${OUT}/${tag}.log" | head -1)"
      continue
    fi

    grep -E "per step|throughput" "${OUT}/${tag}.log" | sed 's/^/  /'

    # 从 json 里抽字段拼 csv。用 python 而非 jq（后者未必装了）。
    python3 - "${OUT}/${tag}.json" "${CSV}" <<'PY'
import json, sys
path, csv_path = sys.argv[1], sys.argv[2]
with open(path) as f:
    d = json.load(f)
row = [
    d["n"], d["block_size"], d["kernel"], d["precision"],
    f'{d["ms_per_step"]:.6f}', f'{d["total_ms"]:.3f}',
    f'{d["particle_steps_per_sec"]:.6e}', f'{d["gflops"]:.3f}',
    f'{d["device_mem_used_mib"]:.1f}', d.get("smem_bytes_per_block", 0),
    f'{d["rel_energy_error"]:.4e}',
    f'{d.get("rel_angular_momentum_error", 0.0):.4e}',
]
with open(csv_path, "a") as f:
    f.write(",".join(str(x) for x in row) + "\n")
PY
  done
done

echo
echo "=== cpu baselines and speedup (N=65536) ==="
# 加速比只在一个有代表性的 N 上测，避免每档都跑一遍 CPU（很慢）。
# 选 65536：作业的进阶要求，且足够大到 GPU 满负载。
#
# 分母是 cpu_openmp（多线程+向量化）。报告里的主结论加速比必须说清是
# 哪个 GPU kernel 对哪档 CPU 基线，否则数字无法复现也无法审查。
ic="${ROOT}/data/uniform_65536.txt"
[ -f "${ic}" ] || "${BIN}/nbody_ic" --preset uniform -n 65536 --out "${ic}"
echo "--- v3 vs cpu baselines ---"
"${BIN}/nbody_sim" \
  --bodies "${ic}" \
  --params "${ROOT}/data/params_bench.txt" \
  --block 128 \
  --out /dev/null \
  --log "${OUT}/speedup_n65536.json" \
  --cpu-bench > "${OUT}/speedup_n65536.log" 2>&1
grep -E "cpu_|per step|gpu speedup|note" "${OUT}/speedup_n65536.log" | sed 's/^/  /'

echo
echo "=== results.csv ==="
cat "${CSV}"

# ---------------------------------------------------------------------------
# 汇总：每个 N 上最优 block，以及相对 N=4096 的规模趋势。
# 这张表是报告里"性能"一节的主表，所以在这里直接算好，
# 避免事后再从几百行 CSV 里手工汇总（容易出错且不可复现）。
# ---------------------------------------------------------------------------
echo
echo "=== best block per N ==="
python3 - "${CSV}" <<'PY'
import csv, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("  (no rows)")
    raise SystemExit

best = {}
for r in rows:
    n = int(r["n"])
    if n not in best or float(r["ms_per_step"]) < float(best[n]["ms_per_step"]):
        best[n] = r

hdr = f'{"N":>8} {"block":>6} {"ms/step":>10} {"GFLOP/s":>10} {"p-steps/s":>12}'
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for n in sorted(best):
    r = best[n]
    print(f'{n:>8} {r["block"]:>6} {float(r["ms_per_step"]):>10.4f} '
          f'{float(r["gflops"]):>10.1f} '
          f'{float(r["particle_steps_per_sec"]):>12.3e}')
PY

echo
echo "artifacts in ${OUT}/"
