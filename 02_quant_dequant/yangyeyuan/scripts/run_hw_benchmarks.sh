#!/usr/bin/env bash
# 硬件/库对照路径基准（加分项）。需要先用 WITH_HW_PATH=1 构建：
#   make BUILD_DIR=build-hw WITH_HW_PATH=1 \
#     CUDA_HOME=~/cuda-root/usr/local/cuda-12.9 CUDA_ARCH=120 -j
# 结果写入 results/hw_path/，每条路径一个目录，内含 hw_path.json/hw_gemm.json。
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bench=${1:-"$project_dir/build-hw/low_precision_cli"}
result_dir="$project_dir/results/hw_path"
fixture_dir="$result_dir/fixtures"

if [ ! -x "$bench" ]; then
  echo "HW 构建不存在：$bench（先用 WITH_HW_PATH=1 构建）" >&2
  exit 1
fi

# 写一组带 hardware_path 的配置；量化参数与 configs/ 下保持一致。
write_config() {
  local path=$1 format=$2 hw=$3
  local block=16
  [ "$format" = "mxfp8" ] && block=32
  cat > "$path" <<EOF
format = "$format"
block_size = $block
scale_mode = "block"
output_type = "fp32"
rounding = "nearest"
target_gpu = "RTX 5060 Ti"
seed = 2026
hardware_path = "$hw"
EOF
}

mkdir -p "$fixture_dir"
write_config "$fixture_dir/mxfp8_cvt.toml" mxfp8 cvt
write_config "$fixture_dir/mxfp8_gemm.toml" mxfp8 gemm
write_config "$fixture_dir/nvfp4_gemm.toml" nvfp4 gemm

# 两种尺寸的 FP32 输入；4096×4096 = 64MB，注意常驻进程占显存时的余量。
for side in ${SIDES:-1024 4096}; do
  python3 "$project_dir/tests/generate_inputs.py" --output "$fixture_dir/$side" \
    --rows "$side" --cols "$side"
  input="$fixture_dir/$side/uniform_fp32.bin"
  "$bench" --input "$input" --config "$fixture_dir/mxfp8_cvt.toml" \
    --output-dir "$result_dir/cvt_$side" \
    --warmup "${WARMUP:-20}" --iters "${ITERS:-200}"
  "$bench" --input "$input" --config "$fixture_dir/mxfp8_gemm.toml" \
    --output-dir "$result_dir/gemm_mxfp8_$side" \
    --warmup "${WARMUP:-20}" --iters "${ITERS:-200}"
  "$bench" --input "$input" --config "$fixture_dir/nvfp4_gemm.toml" \
    --output-dir "$result_dir/gemm_nvfp4_$side" \
    --warmup "${WARMUP:-20}" --iters "${ITERS:-200}"
done

echo "HW benchmark JSON files are under $result_dir"
