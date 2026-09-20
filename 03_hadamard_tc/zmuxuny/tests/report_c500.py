"""Generate the C500 report from checked-in measurements, without rerunning kernels."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/c500"
HAD = ROOT.parent.name == "03_hadamard_tc"


def us(row, key):
    return f"{row[key] * 1000:.2f}" if key in row else "—"


def main():
    validation = json.loads((RESULTS / "correctness.json").read_text())
    report = json.loads((RESULTS / "comparison.json").read_text())
    device = json.loads((RESULTS / "device.json").read_text())
    lines = ["# MetaX C500 适配与性能报告", "",
             "实测日期：2026-09-20（北京时间）。本报告使用同一台 C500 的优化前后数据；NVIDIA 历史结果单独保留。", "",
             "## 环境与复现", "",
             f"单卡 {device['name']}，标称 64GB，{device['multiprocessors']} 个计算单元，原生 wave 为 {device['warp_size']} 线程。",
             "实际环境为 Ubuntu 24.04.1、MACA SDK 3.3.0.15、驱动 3.8.30、Python 3.10.10、NumPy 1.26.4。CPU 配额为 12 核，内存 120GiB。以实测环境为准。",
             "环境原始记录：[environment.txt](results/c500/environment.txt)、[device.json](results/c500/device.json)。", "",
             "```bash", "export MACA_PATH=/opt/maca",
             'export LD_LIBRARY_PATH="$MACA_PATH/lib:${LD_LIBRARY_PATH:-}"',
             '# 本次镜像的 NumPy 位于 /opt/conda；已配置 Python 的环境不需要此行。',
             'export PATH="/opt/conda/bin:$PATH"',
             "make PLATFORM=metax", "mkdir -p results/c500",
             f"python3 tests/validate.py --binary build/metax/{'hadamard' if HAD else 'quantize'} --output results/c500/correctness.json",
             f"python3 tests/benchmark.py --binary build/metax/{'hadamard' if HAD else 'quantize'} --output-dir results/c500 --trials 3 --repeats 100" + (" --materialized-compare" if HAD else ""),
             "python3 tests/profile_metax.py", "```", "",
             "`PLATFORM=nvidia` 仍是默认值；C500 二进制单独放在 `build/metax/`。MACA 使用 cu-bridge 编译及运行时映射，不传 NVIDIA 架构参数。`make native` 是 NVIDIA 原生 FP8 对照，不用于 C500。", "",
             "## 实现", "",
             "- BF16 使用确定性 RNE 位转换，屏蔽 NVIDIA 专用 PTX；MXFP8/NVFP4 保持软件编码和原有文件协议。",
             "- 量化保留 16/32 元素逻辑分组；C500 大张量每线程处理 8 个值，小于等于 65536 元素时使用标量 tile。FP32 全局最大值归约采用每线程 4 元素。",
             "- 反量化按小张量、FP32 输出及 16 位输出分别选择启动配置。调参记录包含所有候选与重复测量：[quantization](results/c500/tune_vector.csv)、[amax](results/c500/tune_max.csv)、[dequantization](results/c500/tune_dequant.csv)。" if not HAD else
             "- 蝶形在 D≤128 时使用 32 线程逻辑分组，更大 D 使用原生 64 线程 wave；矩阵路径用 MACA 原生 FP16/BF16 m16n16k16 计算 H16，其余阶段在 FP32 寄存器和 wave shuffle 中完成，保持 O(D log D)。",
             ]
    if HAD:
        lines += ["- MACA fragment 布局来自实测 SDK 的 `__clang_maca_mma_functions.h`。A 的每线程位置为 `(lane%16, 4*(lane/16)+j)`，B/C 为 `(4*(lane/16)+j, lane%16)`；该实现单独放在 `src/metax_matrix.cuh`。",
                  "- 保留 FP16 稠密 WMMA 对照；FP16/BF16 分解矩阵路径分别测试非融合、全融合，以及 NVFP4 的变换+amax写回后向量量化方案。所有方案都保留中间 dtype 舍入；矩阵输入 fragment 每线程用一次 8 字节向量读取替代四次 16 位读取。",
                  "- 日志沿用 `factorized_tc_*` 字段名，在 C500 上表示 MACA 矩阵单元；默认输出仍为蝶形路径。矩阵路径可用 `--factorized_tc_output`、`--factorized_tc_packed` 保存；NVFP4 两步方案通过 `--materialized_compare 1 --factorized_tc_materialized_packed <文件>` 保存。"]
    lines += ["", "## 正确性", "",
              f"独立 NumPy 参考验证 **{validation['passed']} 个用例全部通过**。结果：[correctness.json](results/c500/correctness.json)。"]
    if HAD:
        lines += [f"其中分解矩阵路径覆盖 {validation['factorized_tc_cases']} 个用例；蝶形 FP16/BF16 最大绝对误差分别为 {validation['max_abs_error_fp16']} / {validation['max_abs_error_bf16']}，矩阵路径分别为 {validation['factorized_tc_max_abs_error_fp16']} / {validation['factorized_tc_max_abs_error_bf16']}。全部低于 0.01 / 0.05。融合与各自非融合结果逐字节一致，NVFP4 两步方案同样逐字节一致。另验证非法维度拒绝。"]
    else:
        lines += ["覆盖 FP32/FP16 输入、FP32/FP16/BF16 输出、两种格式、块/张量缩放、最近偶数/随机舍入、奇数列与尾块、极小值/大值、码本中点及其两侧、65536 元素分派边界；另有 3 个错误文件拒绝测试。"]
    regression = RESULTS / "correctness_nvidia_regression.json"
    if regression.exists():
        r = json.loads(regression.read_text())
        lines += [f"本机 RTX 3060（ARCH=86）回归：{r['passed']} 个用例通过。记录：[NVIDIA regression](results/c500/correctness_nvidia_regression.json)。"]
    lines += ["", "## 性能", "",
              "采用设备事件计时，先预热；每组 3 次独立试验，交替执行优化前后程序，取中位数。小规模每次重复 100 次，大规模重复 30 次。计时包含该流程的全部 GPU 步骤，排除文件 I/O、CPU 参考和主机传输。每次校验输出文件 SHA-256 一致。",
              f"对照基线：{report['baseline']}。二进制校验和、逐次时间和输出哈希见 [comparison.json](results/c500/comparison.json)。", "",
              "![C500 性能对照](results/c500/performance.png)", ""]
    if not HAD:
        lines += ["| 元素数 | 输入 | 格式 | 量化前 μs | 量化后 μs | 加速比 | 反量化后 μs（FP32） |",
                  "|---:|---|---|---:|---:|---:|---:|"]
        for r in report['records']:
            b, a = r['median']['before'], r['median']['after']
            lines.append(f"| {r['rows'] * r['cols']:,} | {r['dtype']} | {r['format']} | {us(b,'quant_ms')} | {us(a,'quant_ms')} | {b['quant_ms']/a['quant_ms']:.2f}× | {us(a,'dequant_ms')} |")
        lines += ["", "大规模组输入均为 128MiB。反量化输出统一为 FP32，因此 FP16 输入组的输出为 256MiB；不能按输入大小直接计算反量化带宽。三种分布与块/张量缩放的完整结果另见 [benchmark.json](results/c500/benchmark.json)。"]
    else:
        lines += ["### 独立变换", "", "| 形状 | dtype | 蝶形前 μs | 蝶形后 μs | 分解矩阵 μs | 矩阵 / 原蝶形加速比 |", "|---|---|---:|---:|---:|---:|"]
        for r in report['records']:
            if r['format'] != 'mxfp8' or r['cols'] not in [64,128,1024] or r['rows']==32:
                continue
            b,a=r['median']['before'],r['median']['after']
            lines.append(f"| {r['rows']}×{r['cols']} | {r['dtype']} | {us(b,'hadamard_ms')} | {us(a,'hadamard_ms')} | {us(a,'factorized_tc_ms')} | {b['hadamard_ms']/a['factorized_tc_ms']:.2f}× |")
        lines += ["", "### 分解矩阵与量化", "", "| 形状 | dtype | 格式 | 非融合 μs | 全融合 μs | 变换+amax后量化 μs |", "|---|---|---|---:|---:|---:|"]
        for r in report['records']:
            if r['cols'] not in [128,1024]:continue
            a=r['median']['after']
            lines.append(f"| {r['rows']}×{r['cols']} | {r['dtype']} | {r['format']} | {us(a,'factorized_tc_unfused_ms')} | {us(a,'factorized_tc_fused_ms')} | {us(a,'factorized_tc_materialized_ms')} |")
        lines += ["", "NVFP4 全融合需要先求全局 amax，再重新计算变换；写回中间结果的方案可以避免第二次变换。MXFP8 和 NVFP4 的最优方案应按表中实际测量选择，不能预设全融合总是最快。",
                  "归一化、随机符号旋转与三种分布的重建误差对照见 [rotation_quality.json](results/c500/rotation_quality.json)。"]
    lines += ["", "## 性能分析工具", "",
              "已使用 MACA 自带 mcTracer 采集设备 kernel、运行时 API 和启动信息。原始 trace 与汇总位于 [profile/summary.json](results/c500/profile/summary.json)。分析器会扰动耗时，性能结论使用上面的未插桩事件计时；trace 只用于观察内核组成、启动配置和执行时间分布。"]
    profile = RESULTS / 'profile/summary.json'
    if profile.exists():
        pr=json.loads(profile.read_text())
        lines += ["", "| trace 用例 | kernel | 次数 | 平均 μs（插桩） |", "|---|---|---:|---:|"]
        for r in pr['kernels']:
            name=r['kernel'].split('(',1)[0].split('lp::',1)[-1]
            lines.append(f"| {r['case']} | `{name}` | {r['count']} | {r['mean_us']:.3f} |")
    lines += ["", "C500 的设备端内存检查本次未运行（镜像未提供可用工具）。NVIDIA 平台既有 Sanitizer 记录保留在 PROFILING.md 所列目录。", ""]
    (ROOT / 'REPORT_C500.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
    from report_final import finalize_platform_report
    finalize_platform_report(ROOT, 'c500')
