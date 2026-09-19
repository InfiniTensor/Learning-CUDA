#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${project_dir}/build"
result_dir="${project_dir}/results/benchmarks"
data_dir="${project_dir}/data/benchmarks"

cmake -S "${project_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel
mkdir -p "${result_dir}" "${data_dir}"

for kind in uniform normal outlier; do
  input="${data_dir}/${kind}_1024x1024_fp32.qdat"
  python3 "${project_dir}/tools/generate_test_data.py" generate \
    --output "${input}" --kind "${kind}" --rows 1024 --cols 1024 \
    --dtype fp32 --seed 20260827
  for format in mxfp8 nvfp4; do
    prefix="${result_dir}/${format}_${kind}_1024x1024"
    "${build_dir}/quant_dequant" --input "${input}" \
      --config "${project_dir}/configs/${format}_block.toml" \
      --quantized "${prefix}.qlow" --output "${prefix}.qdat" \
      --metrics "${prefix}.json" --warmup 10 --iterations 100
  done
done

python3 "${project_dir}/tools/summarize_results.py" "${result_dir}"/*.json \
  --output "${result_dir}/summary.csv"
echo "Benchmark suite completed: ${result_dir}/summary.csv"
