#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${project_dir}/build"
data_dir="${project_dir}/data"

"${project_dir}/scripts/check_env.sh"

cmake -S "${project_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel
ctest --test-dir "${build_dir}" --output-on-failure

mkdir -p "${data_dir}"
python3 "${project_dir}/tools/generate_test_data.py" suite \
    --output-dir "${data_dir}" --rows 64 --cols 65 --dtype fp32 --seed 20260827

echo
echo "Bootstrap completed. Generated data is in ${data_dir}."

