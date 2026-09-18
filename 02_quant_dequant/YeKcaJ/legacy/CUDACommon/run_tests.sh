#!/usr/bin/env bash
# 任一构建、测试或管道中的检查失败就停止；不重新生成冻结 golden。
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
for format in MXFP8 NVFP4; do
  module="$project_dir/CUDA$format"
  cmake -S "$module" -B "$module/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$module/build" -j 4
  mkdir -p "$module/build/tests/results"
  ctest --test-dir "$module/build" --output-on-failure | tee "$module/build/tests/results/ctest.log"
  # 验证固定输入和 golden 未被修改，而不仅是比较本次计算结果。
  (cd "$project_dir/CPU$format" && sha256sum -c tests/golden/SHA256SUMS.txt) \
    | tee "$module/build/tests/results/cpu_hashes.log"
  # 指定工具路径才运行 GPU 检查；未指定表示跳过，不能算作检查通过。
  if [[ -n "${SANITIZER_BIN:-}" ]]; then
    for check in memcheck racecheck synccheck; do
      options=()
      if [[ "$check" == memcheck ]]; then options=(--padding 32 --leak-check full); fi
      (cd "$module/build" && "$SANITIZER_BIN" --tool "$check" "${options[@]}" \
        --error-exitcode 99 --log-file "tests/results/$check.log" \
        ./regression "$project_dir/CPU$format")
      (cd "$module/build" && "$SANITIZER_BIN" --tool "$check" "${options[@]}" \
        --error-exitcode 99 --log-file "tests/results/pipeline_$check.log" \
        ./pipeline --self-test "$project_dir/CPU$format")
    done
  fi
done
