#!/usr/bin/env bash
set -euo pipefail

# C500 上 v0 与 v7 的交错、多输入复测。结果按 RUN_ID 写入独立目录，
# 避免覆盖历史运行；v7 每轮与 v0 的先后顺序交替，降低频率漂移影响。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/metax"
ROUNDS="${ROUNDS:-3}"
WARMUP="${WARMUP:-5}"
REPEATS="${REPEATS:-50}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${BUILD_DIR}/multirun_v0_v7_${RUN_ID}"

bash "${ROOT_DIR}/scripts/build_metax.sh"

mkdir -p "${OUT_DIR}"
for ((round = 1; round <= ROUNDS; round++)); do
  for input in "${ROOT_DIR}"/input/benchmark/*.tensor.bin; do
    stem="$(basename "${input}" .tensor.bin)"
    if ((round % 2 == 0)); then versions=(v7 v0); else versions=(v0 v7); fi
    for version in "${versions[@]}"; do
      if [[ "${version}" == v0 ]]; then executable="${BUILD_DIR}/luxorion_metax_cli";
      else executable="${BUILD_DIR}/luxorion_metax_v7"; fi
      "${executable}" run "${input}" "${ROOT_DIR}/input/quant.cfg" \
        "${OUT_DIR}/${version}_r${round}_${stem}" "${WARMUP}" "${REPEATS}" \
        >"${OUT_DIR}/${version}_r${round}_${stem}.log" 2>&1
    done
  done
done

find "${OUT_DIR}" -name report.json -print | sort | while read -r report; do
  case_id="$(basename "$(dirname "${report}")")"
  version="${case_id%%_r*}"
  stem="${case_id#*_r?_}"
  quant="$(grep '"quantize_kernel_ms"' "${report}" | sed -E 's/.*"median": ([0-9.]+).*/\1/')"
  dequant="$(grep '"dequantize_kernel_ms"' "${report}" | sed -E 's/.*"median": ([0-9.]+).*/\1/')"
  printf '%s,%s,%s,%s\n' "${version}" "${stem}" "${quant}" "${dequant}"
done >"${OUT_DIR}/summary.csv"

awk -F, '{k=$1 SUBSEP $2; q[k]+=$3; d[k]+=$4; n[k]++}
  END {for(k in n){split(k,a,SUBSEP); printf "%s,%s,%.9f,%.9f,%d\n",a[1],a[2],q[k]/n[k],d[k]/n[k],n[k]}}' \
  "${OUT_DIR}/summary.csv" | sort >"${OUT_DIR}/means.csv"

awk -F, '$1=="v0" {q0[$2]=$3; d0[$2]=$4}
  $1=="v7" {q7[$2]=$3; d7[$2]=$4}
  END {for(k in q0){qg+=log(q0[k]/q7[k]); dg+=log(d0[k]/d7[k]); n++}
       printf "geomean_quant_speedup=%.6fx\ngeomean_dequant_speedup=%.6fx\ninputs=%d\n",exp(qg/n),exp(dg/n),n}' \
  "${OUT_DIR}/means.csv" | tee "${OUT_DIR}/aggregate.txt"

echo "results: ${OUT_DIR}"
