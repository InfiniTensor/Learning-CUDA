"""Generate the final NVIDIA report from baseline and endpoint measurements."""

from pathlib import Path
from report_final import read, table, measurements, total_table, final_matrix_table

ROOT = Path(__file__).resolve().parents[1]
HAD = ROOT.parent.name == "03_hadamard_tc"
DETAILS = """## 最终实现与参数选择

BF16 在 Ampere 及更新架构使用原生 RN 转换，输出相邻元素用 `cvt.rn.bf16x2.f32` 打包；CPU 与 sm_75 保留整数位运算实现。MMA 仍采用 H16 小块乘法和 FP32 蝶形分解。普通量化和非融合对照均使用向量化量化与 amax 归约。融合并非每种形状都优于非融合，原始 JSON 同时提供 `factorized_tc_unfused_ms` 与 `factorized_tc_fused_ms`。

[线程块扫描](results/4090d/tuning/tune_mma.csv) 比较 1/2/4/8/16 个 warp。多数尺寸4 warp 配置较好，大尺寸增加 warp 并没有稳定的统一收益，因此默认保持 4 warp。该扫描在软件编码下完成，比较变换输出和 packed 字节，不将噪声级差异用于复杂形状调度。

`--materialized_compare 1` 是 NVFP4 的对照路径：第一次变换同时保存中间张量和 amax，再向量量化，避免第二次 Hadamard，但增加中间写回/读取。每次执行都与同一 MMA 的非融合结果比较 packed 字节；原始 JSON 的 `factorized_tc_materialized_ms` 保存测量。该方案在参数扫描中没有稳定的整体优势，保持为显式对照选项。


## 可选原生 FP8 对照

默认 `make` 仍构建软件 E4M3/E2M1 编码。额外执行 `make native ARCH=89` 可生成独立的 `_native` 程序，要求 CUDA >=12.1 和 sm_89 或更新；两路可以同时存在。原生对照使用 `cvt.rn.satfinite.e4m3x2.f32` 一次转换两个相邻值，低/高字节顺序与 packed 格式一致。随机舍入沿用软件编码，NVFP4 的 E2M1 数据仍由软件编码，只有 E4M3 scale 可用原生转换。日志 `fp8_encoding` 明确区分两路。

指令与架构依据：[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)。这个对照不替代题目 2 的软件实现，也不是 FP8/FP4 Tensor Core 矩阵乘法。

"""


def main():
    baseline = "0a5e40b" if HAD else "73b66de"
    rows = measurements(ROOT, "4090d")
    data = read(ROOT / "results/4090d/tuning/comparison.json")
    text = [
        "# RTX 4090 D 最终性能报告\n",
        "实测日期：2026-09-19。RTX 4090 D 24 GiB、CUDA 12.8、驱动 570.124.06、GCC 13，目标 sm_89，保留 `--fmad=false`。默认程序使用软件 FP8/FP4 编码；原生 E4M3 转换是独立可选对照。\n",
        "## 基线、计时与总收益定义\n",
        f"起始基线固定为 `{baseline}`（对应实验快照 `deba5cb`），最终实现固定为 [设备实验中软件程序的 SHA256](results/4090d/tuning/comparison.json)。总加速比 = 起始基线耗时 / 最终耗时；耗时减少 = 1 − 最终耗时 / 起始基线耗时。\n",
        "两端来自同一台设备、相同形状/dtype/格式的分批归档测量，各含 3 次独立进程，预热 3 次，常规尺寸重复 100 次、128 MiB 重复 30 次，取中位数。GPU 未锁频；这是归档端点的直接比较，并非一次新的基线/最终交替重测。输入 seed=42、标准正态、默认块缩放、nearest。Hadamard 开启归一化、关闭随机符号。\n",
        "设备事件时间包含该流程全部 kernel，NVFP4 包含全局 amax；排除分配、文件 I/O、主机参考与传输。常规尺寸可受缓存影响，128 MiB 输入超过 72 MiB L2。\n",
        "原始数据：[常规尺寸基线](results/4090d/before/benchmark.json)、[128 MiB 基线](results/4090d/extended.json)、[最终实现](results/4090d/tuning/comparison.json)。可用 `python3 tests/report_4090.py` 从这些 JSON 重建本报告。\n",
        "## 总体优化结果\n",
    ]
    if HAD:
        text += [
            "比较完整“变换＋量化”。基线与最终实现各取已测完整路径中的最快方案，并明确列出路径；这体现整个提交的收益，包含算法和执行方式的选择。矩阵与蝶形分别满足误差阈值，融合与各自的非融合输出逐字节一致。\n"
        ]
    else:
        text += [
            "比较完整软件量化流程，反量化另列；不把可选原生 FP8 的收益计入软件实现。\n"
        ]
    text += [total_table([r for r in rows if r["rows"] != 32], HAD)]
    if HAD:
        text += [
            "\n## 最终独立变换：矩阵与蝶形\n",
            "本表只回答最终实现应选哪种变换算法，与上表完整流程的总收益分开。单位 ms。\n",
            final_matrix_table(ROOT, "4090d"),
        ]
        header = [
            "形状 / dtype",
            "格式",
            "蝶形非融合",
            "蝶形融合",
            "矩阵非融合",
            "矩阵融合",
            "矩阵写回+amax",
            "原生 FP8 矩阵融合",
        ]
        rows_final = []
        for r in data["records"]:
            if r["rows"] == 32:
                continue
            m = r["median"]["after"]
            n = r["median"]["native"]
            values = [
                f"{m[k]:.5f}" if m.get(k, 0) > 0 else "—"
                for k in [
                    "unfused_ms",
                    "fused_ms",
                    "factorized_tc_unfused_ms",
                    "factorized_tc_fused_ms",
                    "factorized_tc_materialized_ms",
                ]
            ]
            rows_final.append(
                [
                    f"{r['rows']}×{r['cols']} / {r['dtype']}",
                    r["format"],
                    *values,
                    f"{n['factorized_tc_fused_ms']:.5f}",
                ]
            )
        text += [
            "\n## 最终完整流程与原生转换对照\n",
            "单位 ms。各列包含对应方案的完整成本；选择方案时需显式使用 README 的运行/输出参数。\n",
            table(header, rows_final),
        ]
    else:
        rows_final = []
        for r in rows:
            a = r["after"]
            b = r["before"]
            # MXFP8 large-input baseline writes FP16; final writes FP32.
            # Report final latency but leave the unlike-output baseline blank.
            compatible = r["baseline_dequant_output"] == "fp32"
            rows_final.append(
                [
                    f"{r['rows']}×{r['cols']} / {r['dtype']}",
                    r["format"],
                    f"{b['dequant_ms']:.5f}" if compatible else "—（FP16 输出）",
                    f"{a['dequant_ms']:.5f}",
                    f"{b['dequant_ms']/a['dequant_ms']:.2f}×" if compatible else "—",
                ]
            )
        text += [
            "\n## 反量化最终结果\n",
            "最终统一输出 FP32。MXFP8 的 128 MiB 基线输出 FP16，与最终输出类型不同，因此只报告最终耗时，不计算该项加速比。\n",
            table(
                ["形状 / 输入 dtype", "格式", "基线 ms", "最终 ms", "总加速比"],
                rows_final,
            ),
        ]
        rows_native = []
        for r in data["records"]:
            s = r["median"]["after"]
            n = r["median"]["native"]
            rows_native.append(
                [
                    f"{r['rows']}×{r['cols']} / {r['dtype']}",
                    r["format"],
                    f"{s['quant_ms']:.5f}",
                    f"{n['quant_ms']:.5f}",
                ]
            )
        text += [
            "\n## 最终软件与原生转换对照\n",
            "单位 ms。该表隔离 E4M3 转换实现的差异，不作为另一段累计加速。\n",
            table(["形状 / dtype", "格式", "软件量化", "原生 FP8 对照"], rows_native),
        ]
    text += ["\n" + DETAILS, "## 正确性与工具证据\n"]
    for filename, label in [
        ("correctness.json", "软件实现"),
        ("correctness_native.json", "原生转换对照"),
    ]:
        c = read(ROOT / "results/4090d/tuning" / filename)
        text.append(
            f"- {label}：{c['passed']} 组通过；[记录](results/4090d/tuning/{filename})。"
        )
    count = 36 if HAD else 48
    text += [
        f"\n软件及原生两路共 {count} 次 memcheck/racecheck/synccheck 通过；[软件检查](results/4090d/tuning/software/profiling.json)、[原生检查](results/4090d/tuning/native/profiling.json)。Nsight Systems 时间线与指令证据见 [工具分析](PROFILING.md)；NCU 计数器受宿主机权限限制。\n",
        "当前公共计算代码的 RTX 3060 回归另见 [测试记录](results/ascend/correctness_nvidia_regression.json)。本页性能对应所链接的 4090 D 实验二进制。\n",
        "## 运行与复现\n",
        "按 [README](README.md) 构建软件或原生转换程序，使用 `tests/validate.py` 验证。`tests/benchmark_tuning.py --before <基线程序> --after <最终程序> --native <原生对照程序> --output <结果.json>` 可重新采集同机对照；输入和计时配置应与本页一致。原始环境、命令与二进制哈希见 [环境记录](results/4090d/tuning/environment.json)。\n",
    ]
    (ROOT / "REPORT_4090D.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
