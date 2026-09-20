"""Rebuild the Ascend report and plot from the committed result files."""

import csv
import json
import statistics
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
HAD = ROOT.parent.name == "03_hadamard_tc"
RESULTS = ROOT / "results/ascend"


def main():
    correct = json.loads((RESULTS / "correctness.json").read_text())
    bench = json.loads((RESULTS / "comparison.json").read_text())
    profile = json.loads((RESULTS / "profile/summary.json").read_text())
    sanitize = json.loads((RESULTS / "sanitize/summary.json").read_text())
    records = bench["records"]
    text = [
        "# 昇腾 910B2 适配、验证与性能报告",
        "",
        "实测日期：2026-09-20。训练营 ID：曹泽阳。",
        "",
        "## 环境与运行",
        "",
        "单卡 Ascend 910B2，64GB HBM，ARM64 鲲鹏主机；CANN 9.0、毕昇编译器及驱动 26.1.1。ACL 使用容器内逻辑设备 0，`npu-smi` 显示物理卡 1。AIV 编译目标为 `dav-c220-vec`，Cube 为 `dav-c220-cube`；调度上限分别为 48 和 24 个 block。[完整环境](results/ascend/environment.txt)。",
        "",
        "```bash",
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh",
        "make PLATFORM=ascend",
        "make PLATFORM=ascend test",
        "python3 tests/profile_ascend.py --tool profile",
        "python3 tests/profile_ascend.py --tool sanitize",
        "```",
        "",
        "输入、配置、packed 文件和命令行与其他后端共用。`ascend/` 实现 Ascend C 内核及 ACL 运行时适配；主机侧保留原有文件校验和独立 C++ 参考。NPU 实测不依赖 PyTorch 或预装 MatMul 算子包。",
        "",
        "## 实现与优化",
        "",
    ]
    if HAD:
        text += [
            "- 蝶形路径在 UB 中执行 FP32 计算：步长 1/2/4 使用标量运算，步长 ≥8 使用向量 Add/Sub 和双缓冲，避免覆盖仍需读取的操作数。输入分块 DMA 搬入，每行只写回一次。",
            "- 原生矩阵路径计算 `H_(D/16) ⊗ H16`。每个 Cube tile 处理 64×16 个输入；ND→NZ 搬运、LoadData、FP16/BF16 Mmad 和 Fixpipe 均由本项目调用。H16 权重在每个 core 中复用，尾 tile 按有效行数及 16 行对齐尺寸搬运，保持 FP32 累加。复杂度仍为 O(D log D)。",
            "- Cube 的 FP32 中间结果经 GM 交给 AIV，后者完成剩余蝶形、归一化和最终 dtype 舍入。随机符号通过独立的位操作准备内核处理，其耗时包含在完整矩阵路径内。",
            "- 融合保留中间 FP16/BF16 舍入。MXFP8 在 AIV 蝶形后直接编码；NVFP4 比较“归约后重算蝶形并编码”和“变换、amax、写回合并后再量化”。后者复用已写回的变换，避免重复蝶形。两种方案都包含全局归约的全部开销。",
            "- 量化按整行批量写出 packed data 和 scale，减少每个小块的 DMA 启动；矩阵与向量阶段使用明确的流水线屏障，向量内核结束前恢复掩码。",
            "",
            "Ascend 实现提供原生 H16 分解矩阵路径。其他后端的 O(D²) 稠密 WMMA 对照在本后端不启用；测试根据实际提供的路径验证输出，并强制要求 D≥16 的昇腾用例包含矩阵分解结果。",
        ]
    else:
        text += [
            "- 软件 E4M3/E2M1 编码、E8M0/E4M3 缩放、最近偶数与随机舍入共用数值函数；支持 FP32/FP16 输入及 FP32/FP16/BF16 输出。FP4 两个值合成一个字节，奇数行尾高半字节清零。",
            "- 初始实现逐组读取 GM，并为每个 scale 和 packed 小块发起搬运。当前实现每次 DMA 搬入最多 1024 个元素，在 UB 内处理完整缩放组，批量写出对应 data/scale；分块不跨行，支持 16–1024 间 16 的倍数块长。",
            "- 全局 amax 以 4096 元素分块，使用向量转换、Abs、ReduceMax，再归并各 core 的部分最大值。部分最大值间隔 64 字节，避免标量写回共享缓存行。全次正规数 FP32 tile 保留标量回退，防止向量路径冲零改变缩放。",
            "- MXFP8 使用精确的 2 的幂缩放；NVFP4 最近舍入比较缩放后的精确码本中点，随机舍入保留商和按元素索引生成的随机数。反量化直接读取 GM 中的 codes/scales，并批量 DMA 写回输出。",
            "- 输出使用支持字节长度的 DataCopyPad，避免奇数列和相邻组通过标量字节写入产生缓存行冲突；Scalar、Vector、MTE2/MTE3 间以事件同步。最终归约后恢复向量掩码。",
        ]
    text += [
        "",
        "## 正确性与兼容性",
        "",
        f'完整独立参考测试 **{correct["passed"]} 组通过**，另有 {correct.get("invalid_shape_cases",correct.get("negative_file_cases"))} 组非法输入检查。[NPU 结果](results/ascend/correctness.json)、[RTX 3060 公共代码回归](results/ascend/correctness_nvidia_regression.json)。',
        "",
        "两题的共享数值函数分别通过 **49,152 项哈希、65,536 项乘法、1,048,576 项除法**的主机/设备逐位对照，覆盖符号、指数边界、正规数、次正规数和舍入边界。[原始日志](results/ascend/numeric.log)。",
    ]
    if HAD:
        text += [
            "",
            f'其中 **{correct["factorized_tc_cases"]} 组**覆盖 FP16/BF16 原生矩阵分解；最大绝对误差分别为 **{correct["factorized_tc_max_abs_error_fp16"]} / {correct["factorized_tc_max_abs_error_bf16"]}**，低于 0.01 / 0.05。每个后端自己的融合/非融合，以及 NVFP4 写回方案均逐字节比较 packed data、scale 和 global scale。',
        ]
    text += [
        "",
        "## 同机性能对照",
        "",
        "基线为[正确性通过的初始 Ascend C 移植](results/ascend/correctness_baseline.json)。基线/当前程序交替运行，每项 3 个独立进程试验取中位数；每条完整流程先预热 3 次，中小输入重复 10 次，128MiB 输入重复 3 次。使用 ACL 事件计时，包含全部 NPU kernel，排除分配、文件 I/O、主机参考与传输。计时保留启动提交间隔，不将其解释为单条硬件指令耗时。每次比较 packed 文件 SHA256。输入为 seed=42 的标准正态数据，最近偶数舍入，默认块缩放。反量化输出固定为 FP32。",
        "",
        "[完整逐次记录及可执行文件/动态库 SHA256](results/ascend/comparison.json)。基线及全部可执行文件另存于完整实验备份；正式代码、构建日志和源码校验清单随提交提供。",
        "",
        "![性能对照](results/ascend/performance.png)",
        "",
    ]
    if not HAD:
        text += [
            "| 输入 MiB | dtype / 格式 | 初始量化 μs | 当前量化 μs | 加速比 | 初始反量化 μs | 当前反量化 μs |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        for r in records:
            b, c = r["median"]["before"], r["median"]["after"]
            text.append(
                f'| {r["input_bytes"]/2**20:g} | {r["dtype"]} / {r["format"]} | {b["quant_ms"]*1000:.2f} | {c["quant_ms"]*1000:.2f} | {b["quant_ms"]/c["quant_ms"]:.2f}× | {b["dequant_ms"]*1000:.2f} | {c["dequant_ms"]*1000:.2f} |'
            )
        text += [
            "",
            "NVFP4 量化计时包含全局 amax。反量化统一输出 FP32，因此 128MiB FP16 输入对应 256MiB 输出。不同硬件之间不据此表计算加速比；小尺寸启动开销和未改善项目均保留在原始记录。反量化的显式 UB 预取候选较慢，正式版选择原来的直接 GM 读取；[候选对照](results/ascend/comparison_dma_dequant_candidate.json) 保留该判断的测量依据。",
        ]
    else:
        text += [
            "| 输入 MiB / dtype | 初始蝶形 μs | 当前蝶形 μs | 初始矩阵 μs | 当前矩阵 μs |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in records:
            if r["format"] != "mxfp8":
                continue
            b, c = r["median"]["before"], r["median"]["after"]
            text.append(
                f'| {r["input_bytes"]/2**20:g} / {r["dtype"]} | {b["hadamard_ms"]*1000:.2f} | {c["hadamard_ms"]*1000:.2f} | {b["factorized_tc_ms"]*1000:.2f} | {c["factorized_tc_ms"]*1000:.2f} |'
            )
        text += [
            "",
            "矩阵列包含 Cube、FP32 中间结果交接和 AIV 剩余蝶形，不能与仅计 Mmad 的时间比较。大尺寸完整量化流程如下：",
            "",
            "| dtype / 格式 | 蝶形非融合 μs | 蝶形融合 μs | 矩阵非融合 μs | 矩阵融合 μs | 矩阵写回+amax μs |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in records:
            if r["input_bytes"] != 128 * 2**20:
                continue
            c = r["median"]["after"]
            v = c.get("factorized_tc_materialized_ms")
            text.append(
                f'| {r["dtype"]} / {r["format"]} | {c["unfused_ms"]*1000:.2f} | {c["fused_ms"]*1000:.2f} | {c["factorized_tc_unfused_ms"]*1000:.2f} | {c["factorized_tc_fused_ms"]*1000:.2f} | '
                + (f"{v*1000:.2f}" if v is not None else "—")
                + " |"
            )
        text += [
            "",
            "保留各路径供按尺寸选择；是否融合、是否使用矩阵指令都以完整流程实测为依据。FP32 中间结果的 GM 交接和软件编码仍有优化空间。",
        ]
    text += [
        "",
        "## Profiler 与 Sanitizer",
        "",
        f'msprof 采集 {len(profile["records"])} 组格式/dtype 用例，导出任务时间与 PipeUtilization 计数器。采样与性能基准独立执行。[原始 CSV 与采样状态](results/ascend/profile/summary.json)。下表取 NVFP4/FP16 用例：65×1024，默认块缩放。',
        "",
        "| kernel | 次数 | 平均 μs | 平均 AIV Scalar 比例 | 平均 AIC MAC 比例 |",
        "|---|---:|---:|---:|---:|",
    ]
    groups = {}
    for file in (RESULTS / "profile/nvfp4_fp16").rglob("op_summary*.csv"):
        with file.open() as f:
            for row in csv.DictReader(f):
                groups.setdefault(row["Op Name"], []).append(row)
    for name, rs in groups.items():
        mean = lambda key: statistics.mean(float(r.get(key, 0) or 0) for r in rs)
        text.append(
            f'| {name} | {len(rs)} | {mean("Task Duration(us)"):.3f} | {mean("aiv_scalar_ratio"):.1%} | {mean("aic_mac_ratio"):.1%} |'
        )
    quant_rows = groups.get("ascend_quant", [])
    if quant_rows:
        scalar = statistics.mean(float(r["aiv_scalar_ratio"]) for r in quant_rows)
        text += [
            "",
            f"量化内核的 AIV Scalar 比例为 {scalar:.1%}；结合软件码本、缩放与逐元素舍入实现，当前优化重点仍在减少标量指令与 UB 标量访问。该比率属于所采样输入，不外推为所有尺寸的利用率。",
        ]
    if HAD:
        text += [
            "",
            "Cube 的 MAC 计数器与原生 Mmad 实现共同核验矩阵路径已执行。小 tile 的搬运、流水线屏障，以及 Cube→GM→AIV 交接均有开销，因此报告比较整个变换和量化流程。",
        ]
    passed = sum(r["status"] == "PASS_BASIC" for r in sanitize["records"])
    text += [
        "",
        "上述流水线比率来自工具，可能与其他流水线重叠，不能相加当作总时间分解。小规模采样用于核验执行路径；大尺寸结论使用独立基准。",
        "",
        f'mssanitizer 基础模式共 **{passed}/{len(sanitize["records"])} 次**满足：进程退出为零、实际 kernel 检查开始与完成次数一致、全部明确无错误且无警告。该平台实验实际执行基础 memcheck；racecheck/initcheck/synccheck 要求源码插桩，当前未能执行，不计作通过。[检查汇总及日志](results/ascend/sanitize/summary.json)。',
        "",
        "完整 `--cce-enable-sanitizer` 源码插桩未通过本镜像的构建/运行验证，不能把基础模式结果表述为全量插桩通过。直接编译及最小样例触发毕昇后端 FrameIndex 错误；分阶段实验超时，原始记录见 [工具限制](results/ascend/tools_limitations/sanitizer_build_debug.log)。基础模式发现的向量掩码恢复警告已修复并复测。",
        "",
        "## 误差实验与复核材料",
        "",
        "常规实验使用每项 3 次试验、每次重复 5 次。[benchmark.json](results/ascend/benchmark.json) 保留误差、压缩率与逐次时间。"
        + (
            "矩阵实验覆盖 64–1024 维；[rotation_quality.json](results/ascend/rotation_quality.json) 在原坐标系比较直接量化和随机旋转后量化的重建 MSE。"
            if HAD
            else "量化实验覆盖均匀、正态、离群点分布，两种输入精度、两种格式与块/张量缩放。"
        ),
        "",
        "[源码 SHA256](results/ascend/source_sha256.json)、[构建日志](results/ascend/build_final.log)、[使用说明](README.md)、[工具复现](PROFILING.md)。其他平台的归档数据对应各自原始版本；当前公共代码完整回归设备为 Ascend 910B2 与 RTX 3060。",
        "",
        "API 依据：[Ascend C DataCopyPad API](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_0265.html)、[msSanitizer 全量检查编译配置](https://github.com/Ascend/mssanitizer/blob/master/docs/zh/user_guide/compile_option_config.md)。实测编译使用服务器安装的 CANN 9.0 头文件。",
        "",
    ]
    executable = "hadamard" if HAD else "quantize"
    text += [
        "## 基线复现",
        "",
        "[还原补丁](results/ascend/baseline_restore.patch) 在本题目录的独立副本中使用：",
        "",
        "```bash",
        "baseline_dir=$(mktemp -d)",
        'cp -r ascend include src tests Makefile "$baseline_dir/"',
        'patch -d "$baseline_dir" -p1 < results/ascend/baseline_restore.patch',
        'make -C "$baseline_dir" PLATFORM=ascend',
        f'python3 tests/benchmark_ascend.py --before "$baseline_dir/build/ascend/{executable}" --after build/ascend/{executable} --large',
        "```",
        "",
        (
            "重建的 Hadamard 主程序、libcamp.so 和 libcube.so 均与实测基线 SHA256 相同。"
            if HAD
            else "重建的量化 libcamp.so 与实测基线 SHA256 相同；主机可执行文件哈希不同，本报告使用归档的原始主程序计时。"
        )
        + "[重建哈希核对](results/ascend/baseline_rebuild.json)。",
        "",
    ]
    headline = next(
        r
        for r in records
        if r["input_bytes"] == 128 * 2**20
        and r["dtype"] == "fp16"
        and r["format"] == ("mxfp8" if HAD else "nvfp4")
    )
    b, c = headline["median"]["before"], headline["median"]["after"]
    if HAD:
        outcome = f"128MiB FP16 独立变换：初始蝶形 {b['hadamard_ms']:.3f} ms，当前蝶形 {c['hadamard_ms']:.3f} ms，当前 Cube 分解 {c['factorized_tc_ms']:.3f} ms。"
    else:
        outcome = f"128MiB FP16/NVFP4 完整量化从 {b['quant_ms']:.3f} ms 降至 {c['quant_ms']:.3f} ms，为初始正确移植的 {b['quant_ms']/c['quant_ms']:.2f}×；同表保留反量化的改善或回退。"
    text[4:4] = [
        f"完成原生 Ascend C 实现，{correct['passed']} 组正确性测试通过；基础 memcheck 4 组和 msprof 4 组均完成。"
        + outcome,
        "",
    ]
    (ROOT / "REPORT_ASCEND.md").write_text("\n".join(text))
    large = [
        r
        for r in records
        if r["input_bytes"] == 128 * 2**20 and (not HAD or r["format"] == "mxfp8")
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    xs = list(range(len(large)))
    width = 0.24 if HAD else 0.34
    if HAD:
        bars = [
            ("Initial butterfly", "before", "hadamard_ms"),
            ("Current butterfly", "after", "hadamard_ms"),
            ("Current Cube + vector", "after", "factorized_tc_ms"),
        ]
    else:
        bars = [
            ("Initial port", "before", "quant_ms"),
            ("Current", "after", "quant_ms"),
        ]
    for j, (label, version, key) in enumerate(bars):
        vals = [r["median"][version][key] for r in large]
        bc = ax.bar(
            [x + (j - (len(bars) - 1) / 2) * width for x in xs],
            vals,
            width,
            label=label,
            color=["#8193a7", "#247d83", "#d59b46"][j],
        )
        ax.bar_label(bc, fmt="%.2f", fontsize=9, padding=3)
    ax.set_xticks(
        xs,
        [
            r["dtype"].upper() + (" / " + r["format"].upper() if not HAD else "")
            for r in large
        ],
    )
    ax.set_ylabel("Complete pipeline time (ms)")
    ax.set_title("Ascend 910B2 | 128 MiB input | median of 3 interleaved trials")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    fig.tight_layout()
    fig.savefig(RESULTS / "performance.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
    from report_final import finalize_platform_report
    finalize_platform_report(ROOT, 'ascend')
