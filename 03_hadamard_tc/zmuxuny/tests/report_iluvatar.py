"""Rebuild the MR-V100 report from committed measurements and native traces."""
import json
from pathlib import Path
import re
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/iluvatar"
HAD = ROOT.parent.name == "03_hadamard_tc"


def read(name):
    return json.loads((RESULTS / name).read_text())


def us(metrics, key):
    return f"{metrics[key] * 1000:.2f}" if key in metrics else "—"


def summarize_profiles():
    records = []
    trace = read("profile/summary_trace.json")
    for run in trace["runs"]:
        file = RESULTS / "profile" / run["log"]
        with sqlite3.connect(file.parent / "trace.ixsys") as db:
            # IXplorer stores durations in ns. Cross-checked against the CLI
            # summary (probe: 4738 ns == 4.738 us), not Chrome trace conventions.
            query = """SELECT n.name, count(*), avg(k.dur)/1000.0,
                       g.name, b.name, k.regs, k.staticSharedMemory
                       FROM cuda_kernel_data k
                       JOIN function_name_map n ON n.id=k.name_id
                       JOIN function_size_map g ON g.id=k.grid_id
                       JOIN function_size_map b ON b.id=k.block_id
                       GROUP BY n.name, k.grid_id, k.block_id, k.regs, k.staticSharedMemory"""
            for name, count, mean, grid, block, regs, shared in db.execute(query):
                records.append(dict(case=run["case"], kernel=name, count=count, mean_us=mean,
                                    grid=grid, block=block, registers=regs, shared_bytes=shared,
                                    trace=str((file.parent / "trace.ixsys").relative_to(RESULTS))))
    counters = []
    for run in read("profile/summary_counters.json")["runs"]:
        log = (RESULTS / "profile" / run["log"]).read_text()
        metrics = {}
        for line in log.splitlines():
            match = re.match(r"\s*(TCU Efficiency|Achieved Occupancy|Theoretical Occupancy|Global Memory Load Throughput|Global Memory Store Throughput|\+Multi-Lane Matrix Instructions)\s+([\d.,]+)\s+(.*?)\s*$", line)
            if match:
                metrics[match[1]] = match[2] + " " + match[3]
        if not metrics:
            raise ValueError(f"Missing counters: {run['log']}")
        counters.append(dict(case=run["case"], stage=run["tool"],
                             kernel_filter=run["command"][run["command"].index("--kernel-name") + 1],
                             metrics=metrics, log=run["log"]))
    result = dict(kernels=records, counters=counters)
    (RESULTS / "profile/analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    validation, comparison, device = read("correctness.json"), read("comparison.json"), read("device.json")
    profiles = summarize_profiles()
    sanitizer = read("profile/summary_sanitizer.json")
    assert all(run["passed"] for run in sanitizer["runs"])
    binary = "hadamard" if HAD else "quantize"
    lines = ["# Iluvatar 智铠 100（MR-V100）适配与性能报告", "",
             "实测日期：2026-09-20。训练营 ID：曹泽阳；提交目录：`zmuxuny`。", "",
             "## 环境与复现", "",
             f"单卡 {device['name']}，32GiB 显存，{device['multiprocessors']} 个计算单元，原生 warp 为 {device['warp_size']} 线程。",
             "Ubuntu 24.04.4、CoreX SDK / 驱动 4.4.0、Python 3.12、NumPy 1.26.4；CPU 配额 12 核，内存 32GiB。使用 CoreX clang 的 `-x ivcore --cuda-gpu-arch=ivcore11 -O3 -ffp-contract=off`。",
             "[环境原始记录](results/iluvatar/environment.txt) · [设备属性](results/iluvatar/device.json)。", "",
             "```bash", "export COREX_PATH=/usr/local/corex",
             'export LD_LIBRARY_PATH="$COREX_PATH/lib64:${LD_LIBRARY_PATH:-}"',
             "make PLATFORM=iluvatar", "make PLATFORM=iluvatar numeric-test",
             f"python3 tests/validate.py --binary build/iluvatar/{binary} --output results/iluvatar/correctness.json",
             f"python3 tests/benchmark.py --binary build/iluvatar/{binary} --output-dir results/iluvatar --trials 3 --repeats 100" + (" --materialized-compare" if HAD else ""),
             "python3 tests/profile_iluvatar.py", "python3 tests/report_iluvatar.py", "python3 tests/plot_iluvatar.py", "```", "",
             "默认 `PLATFORM=nvidia`；天数二进制单独存放在 `build/iluvatar/`。软件 FP8/FP4 编码及文件协议与其他后端相同，NVIDIA 专用 PTX 不进入 CoreX 编译路径。", "",
             "## 设计", "",
             "- 量化保留 32/16 元素逻辑缩放组；FP32/MXFP8 使用每线程 2 元素，较大 FP16/NVFP4 使用 8 元素。小张量独立选择启动配置，65536 元素分派边界有测试覆盖。",
             "- 全局 amax 对 FP32 使用标量连续访问；16 位输入使用 4 元素向量及受限网格。计时包含清零、全局归约及最终量化。",
             "- BF16 转换、负零符号和整数钳位同时支持 host/device。CoreX 4.4 在优化 lane 索引哈希时可错误复用移位结果；仅该哈希的设备局部变量使用 `volatile`，并以 49,152 次逐项测试覆盖线程边界、种子和超过 2³² 的索引。未改变随机序列或舍入语义。"]
    if HAD:
        lines += ["- 原生 FP16/BF16 矩阵指令实现 H16，其余因子以 FP32 蝶形完成，复杂度 O(D log D)。实现位于 [iluvatar_matrix.cuh](src/iluvatar_matrix.cuh)，并非用蝶形冒充矩阵路径。",
                  "- 令 `g=lane/16, c=lane%16`。输入 A 每线程元素位于 `(2g+j%2+8⌊j/2⌋, c)`，B 使用相同的 K 索引；FP32 累加器位于 `(g+4j, c)`。映射依据 CoreX 4.4 的 `crt/iluvatar_mma.hpp`；与 NVIDIA/MACA 的 fragment 分开实现，避免直接复用不兼容布局。",
                  "- 蝶形按阶段选择 32/64 线程组，amax 预遍历使用更多行/CTA，D≥512 的输出阶段使用两个 64 线程组以降低寄存器压力。稠密矩阵对照使用补偿求和，避免长串 FP32 加法在未归一化 FP16 舍入中点处产生大于阈值的误差。",
                  "- 融合前保留中间 FP16/BF16 舍入，packed 结果与各自非融合流程逐字节一致。另提供 NVFP4 的变换+amax 写回、再量化方案。默认输出仍为蝶形；矩阵输出使用 `--factorized_tc_output` / `--factorized_tc_packed`，两步量化使用 `--materialized_compare 1 --factorized_tc_materialized_packed <文件>`。",
                  "- 矩阵与蝶形启动扫描：[tune_mma.csv](results/iluvatar/tune_mma.csv)、[tune_butterfly.csv](results/iluvatar/tune_butterfly.csv)。"]
    else:
        lines += ["- 向量宽度、线程数、归约网格扫描保留全部候选：[量化](results/iluvatar/tune_vector.csv)、[amax](results/iluvatar/tune_max.csv)、[反量化](results/iluvatar/tune_dequant.csv)。候选先校验输出一致，再比较未插桩计时。"]
    lines += ["", "## 验证", "",
              f"独立 NumPy 参考：**{validation['passed']} 组全部通过**，见 [correctness.json](results/iluvatar/correctness.json)。"]
    if HAD:
        lines += [f"其中 {validation['factorized_tc_cases']} 组覆盖分解矩阵路径；FP16/BF16 最大绝对误差为 {validation['factorized_tc_max_abs_error_fp16']} / {validation['factorized_tc_max_abs_error_bf16']}，分别低于 0.01 / 0.05。覆盖尾行、1–1024 维、归一化/随机符号、随机舍入与 NVFP4 两步方案；另验证非法尺寸拒绝。"]
    else:
        lines += ["覆盖两种格式、FP32/FP16 输入、三种输出 dtype、块/张量缩放、最近偶数/随机舍入、中点/ULP、奇数列/尾块及非法文件；另有 3 组文件错误检查。"]
    regression = read("correctness_nvidia_regression.json")
    lines += [f"RTX 3060 / sm_86 完整回归 **{regression['passed']} 组通过**：[记录](results/iluvatar/correctness_nvidia_regression.json)。设备/主机随机哈希的精确比较见 [numeric.log](results/iluvatar/numeric.log)。",
              f"设备检查：**{len(sanitizer['runs'])} 次 ixsan 全部通过**，涵盖 memcheck、racecheck、initcheck；逐次命令、退出码、零错误汇总和二进制 SHA256 见 [检查汇总](results/iluvatar/profile/summary_sanitizer.json)。该工具未提供 synccheck。", "",
              "## 性能对照", "",
              "在同一张 MR-V100 上交替执行基线与当前版本，每项 3 次独立试验，取设备事件计时的中位数。预热 3 次，小规模重复 100 次，大规模重复 30 次。包含完整 GPU 流程，排除文件读写、主机传输和 CPU 参考。每次验证 packed SHA256 一致。",
              f"基线：{comparison['baseline']}。完整逐次时间与二进制哈希见 [comparison.json](results/iluvatar/comparison.json)；[initial_port.patch](results/iluvatar/initial_port.patch) 可在本目录副本中通过 `patch -p1 < results/iluvatar/initial_port.patch` 还原正确性已通过的初始移植源码，再执行 `make -B PLATFORM=iluvatar`。",
              "基线和当前程序可用 `tests/benchmark_tuning.py --before <基线> --after <当前> --output <JSON> --trials 3` 重新比较。补丁包含该基线所需的构建和平台兼容代码。", "",
              "![MR-V100 性能对照](results/iluvatar/performance.png)", ""]
    if not HAD:
        lines += ["| 元素数 | 输入 | 格式 | 量化基线 μs | 量化当前 μs | 加速比 | 当前反量化 μs（FP32） |",
                  "|---:|---|---|---:|---:|---:|---:|"]
        for row in comparison["records"]:
            before, after = row["median"]["before"], row["median"]["after"]
            lines.append(f"| {row['rows'] * row['cols']:,} | {row['dtype']} | {row['format']} | {us(before,'quant_ms')} | {us(after,'quant_ms')} | {before['quant_ms']/after['quant_ms']:.2f}× | {us(after,'dequant_ms')} |")
        lines += ["", "大尺寸输入均为 128MiB；反量化统一输出 FP32，因此 FP16 输入组产生 256MiB 输出。FP16/MXFP8 大尺寸主要保留原有配置，收益有限；按格式分别报告，不能套用 NVFP4 加速比。三种分布及块/张量缩放的结果见 [benchmark.json](results/iluvatar/benchmark.json)。"]
    else:
        lines += ["### 独立变换", "", "| 形状 | dtype | 初始蝶形 μs | 当前蝶形 μs | 当前矩阵 μs | 当前矩阵 / 初始蝶形加速比 |", "|---|---|---:|---:|---:|---:|"]
        for row in comparison["records"]:
            if row["format"] != "mxfp8" or row["rows"] == 32: continue
            before, after = row["median"]["before"], row["median"]["after"]
            lines.append(f"| {row['rows']}×{row['cols']} | {row['dtype']} | {us(before,'hadamard_ms')} | {us(after,'hadamard_ms')} | {us(after,'factorized_tc_ms')} | {before['hadamard_ms']/after['factorized_tc_ms']:.2f}× |")
        lines += ["", "### 矩阵变换与量化", "", "| 形状 | dtype | 格式 | 非融合 μs | 全融合 μs | 变换+amax 写回后量化 μs |", "|---|---|---|---:|---:|---:|"]
        for row in comparison["records"]:
            if row["cols"] not in [128,1024]: continue
            after = row["median"]["after"]
            lines.append(f"| {row['rows']}×{row['cols']} | {row['dtype']} | {row['format']} | {us(after,'factorized_tc_unfused_ms')} | {us(after,'factorized_tc_fused_ms')} | {us(after,'factorized_tc_materialized_ms')} |")
        lines += ["", "NVFP4 全融合需要先求全局 amax，再重新变换；两步方案写回中间结果，避免第二次变换。最优路径按尺寸和格式比较，不能由 kernel 数量直接判断。表中的矩阵 / 初始蝶形加速比是算法路径对照，并非同一矩阵 kernel 的迭代加速比；后者原始数据也保留在 comparison.json。",
                  "正常分布和离群值分布的旋转重建误差见 [rotation_quality.json](results/iluvatar/rotation_quality.json)。"]
    lines += ["", "## Profiler", "",
              "ixsys 原生时间线与 ixkn 硬件计数器均已采集。以下为插桩统计，不能与上面未插桩基准直接比较：[时间线](results/iluvatar/profile/summary_trace.json)、[计数器](results/iluvatar/profile/summary_counters.json)、[机器可读分析](results/iluvatar/profile/analysis.json)。", "",
              "| 用例 | kernel | 次数 | 平均 μs（插桩） |", "|---|---|---:|---:|"]
    for kernel in profiles["kernels"]:
        if HAD and "hadamard_mma_kernel" not in kernel["kernel"]: continue
        name = kernel["kernel"].split("(",1)[0].split("lp::",1)[-1]
        lines.append(f"| {kernel['case']} | `{name}` | {kernel['count']} | {kernel['mean_us']:.3f} |")
    lines += ["", "| 计数器用例 / kernel | TCU 效率 | 实测占用率 | 矩阵指令数 |", "|---|---:|---:|---:|"]
    for counter in profiles["counters"]:
        metrics = counter["metrics"]
        lines.append(f"| {counter['case']} / {counter['stage']} | {metrics.get('TCU Efficiency','—')} | {metrics.get('Achieved Occupancy','—')} | {metrics.get('+Multi-Lane Matrix Instructions','—')} |")
    lines += ["", "计数器结果来自单次 kernel replay；指令计数用于核实矩阵路径，占用率用于观察当前启动配置，不单独作为速度判断依据。", ""]
    if HAD:
        lines += ["大维度样本为 8192×1024：独立变换与每次融合变换均记录到 32,768 条矩阵指令，与 256 元素/H16 tile 的分解一致。NVFP4 全融合的 amax 预遍历和最终写出各执行一次变换；写回方案在一次变换中同时得到输出和 amax。软件编码、寄存器中间值及写出方式也不同，因此减少 kernel 数量并不保证降低总时间。以上完整流程计时支持大尺寸选择非融合 MXFP8 / 两步 NVFP4 的结论。", ""]
    (ROOT / "REPORT_ILUVATAR.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
    from report_final import finalize_platform_report
    finalize_platform_report(ROOT, 'iluvatar')
