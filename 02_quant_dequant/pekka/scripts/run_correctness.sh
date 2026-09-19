#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${project_dir}/build"
result_dir="${project_dir}/results/correctness"
data_dir="${project_dir}/data/correctness"

cmake -S "${project_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel
ctest --test-dir "${build_dir}" --output-on-failure

mkdir -p "${result_dir}" "${data_dir}"
for size in 1 15 16 17 31 32 33; do
  python3 "${project_dir}/tools/generate_test_data.py" generate \
    --output "${data_dir}/normal_1x${size}.qdat" --kind normal \
    --rows 1 --cols "${size}" --dtype fp32 --seed 20260827
  for format in mxfp8 nvfp4; do
    prefix="${result_dir}/${format}_normal_1x${size}"
    "${build_dir}/quant_dequant" \
      --input "${data_dir}/normal_1x${size}.qdat" \
      --config "${project_dir}/configs/${format}_block.toml" \
      --quantized "${prefix}.qlow" --output "${prefix}.qdat" \
      --metrics "${prefix}.json" --warmup 1 --iterations 2
  done
done

echo "Correctness suite completed: ${result_dir}"
