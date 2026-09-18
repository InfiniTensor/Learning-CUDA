#!/usr/bin/env python3
"""用 nsys 采集一个固定规模；保留报告、统计、环境和采集消息。"""

import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

# 默认量化 kernel 名，必须与 runtime/workspace.cu 的启动代码一致。
# 测试也从这里派生 mock 数据，避免测试与实现各写一份而悄悄失效。
DEFAULT_KERNEL = {
    "mxfp8": "pipeline::mxfp8_quantize_vectorized_kernel",
    "nvfp4": "pipeline::nvfp4_quantize_fused_kernel",
}


# 串行采集两种格式的固定 4M benchmark，定位 kernel、API 和传输耗时。
# 报告包含预热、内部对照、默认路径和 host_api 循环，并非只有一次量化。
def main():
    # ===== 1. 新建结果目录并记录 nsys 版本 =====
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "version.txt").write_text(
        subprocess.check_output(["nsys", "--version"], text=True)
    )
    # ===== 2. 逐格式采集 CUDA 时间线 =====
    for fmt in ("mxfp8", "nvfp4"):
        prefix = directory / fmt
        binary = ROOT / "build" / ("pipeline_" + fmt)
        # 关闭 CPU 采样/上下文切换采集；benchmark 正式重复数为 5，预热另外执行。
        cmd = [
            "nsys",
            "profile",
            "--trace=cuda",
            "--sample=none",
            "--cpuctxsw=none",
            "-o",
            str(prefix),
            str(binary),
            "--benchmark",
            str(4 << 20),
            "5",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        (directory / f"{fmt}_capture.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        # ===== 3. 导出 kernel、主机 CUDA API、GPU 内存传输三类统计 =====
        result = subprocess.run(
            [
                "nsys",
                "stats",
                "--report",
                "cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum",
                str(prefix) + ".nsys-rep",
            ],
            capture_output=True,
            text=True,
        )
        (directory / f"{fmt}_stats.txt").write_text(result.stdout + result.stderr)
        result.check_returncode()
        # 返回码成功不代表采到了 GPU 数据；还需检查默认量化 kernel 是否出现在报告中。
        # 名字必须与 runtime/workspace.cu 的默认路径一致：MXFP8 已改为向量化 kernel，
        # NVFP4 仍是两级归约 + 融合 kernel，两者不再共用一个后缀。
        expected = DEFAULT_KERNEL[fmt]
        if expected not in result.stdout or "SKIPPED" in result.stdout:
            raise RuntimeError(f"missing profiler data for {fmt}; inspect capture log")
        print(f"{fmt}: kernel/API/memory data collected", flush=True)


if __name__ == "__main__":
    main()
