#!/usr/bin/env python3
"""
file_index.py —— final_version/ 逐文件用途说明

用法：
    python3 tools/file_index.py                 # 平面列表（路径 --- 说明）
    python3 tools/file_index.py --group         # 按目录分组
    python3 tools/file_index.py > FILE_INDEX.md # 导出

输出格式（平面模式）：
    <路径> --- <一句话用途>

注意：若以下文件仍然存在，说明删除尚未执行——
    include/nb_platform*.h  scripts/build_platform.sh
它们是已废弃的国产适配层残留，不被任何源文件引用，本索引不描述其内容。

本工程不包含 tests/ 目录，也不启用 CTest：正确性门禁全在 scripts/validate.sh
（8 段 16 项），其中包括对二进制粒子主序布局的字节级交叉判定（段 1）。
"""

INDEX = [
    # ---- 构建入口 ----------------------------------------------------------------
    "CMakeLists.txt",
    "双架构构建配置（sm_61/sm_89），MSVC 与 gcc 双平台，CUDA 语言 + C++17，"
    "三档 CPU 基线（naive/scalar/OpenMP），-rdc ON。",

    # ---- 文档（两份） -----------------------------------------------------------
    "README.md",
    "项目入口文档：构建、运行、CLI 参数、数据格式、验证流程、性能速查、可视化。",

    "n_bodyversion.md",
    "最终提交报告：自包含——实现思路、数值积分、优化方法、正确性验证、"
    "性能分析、nsys、未来提升、逐文件索引。",

    ".clang-format",
    "代码风格配置（BasedOnStyle: LLVM 基础上微调）。clang-format -i 或 CLion "
    "Ctrl+Alt+L 直接生效；不影响构建与运行。",

    # ---- 数据：初值与参数 -------------------------------------------------------
    "data/params_bench.txt",
    "性能扫参用参数表：dt=1e-3, 1000 步, record_interval=100（R=11，不适合动画）。",

    "data/params_cluster.txt",
    "星团演化用参数表：dt=2e-3, 2000 步, record_interval=10（R=201，适合 3D 动画）。",

    "data/params_collision.txt",
    "双球碰撞用参数表：dt=2e-3, 150000 步, record_interval=750, softening=0.005。",

    "data/params_solar.txt",
    "太阳系初值用参数表：dt=86.4ks, 3650 步, record_interval=10, softening=0。",

    "data/params_two_body.txt",
    "二体圆轨道用参数表：dt=1e-3, 44429 步（≈T）, record_interval=50, softening=0。",

    "data/params_two_body_euler.txt",
    "二体 Euler 对照用参数表：dt=5e-4, 88858 步, record_interval=100。",

    # ---- include/：头文件 ---------------------------------------------------------
    "include/nbody_types.h",
    "全局精度定义（real = float / double，由 NBODY_FP64 控制）、"
    "SoA 粒子数据布局（DeviceBodies）、real4 打包类型与 make_real4() 工厂、"
    "FLOP 计数约定（20 flop / 对相互作用）。",

    "include/cpu_kernel_inl.h",
    "CPU 三档基线的共享内核实现（模板头文件）："
    "力计算的 O(N²) 内循环、inv_r 求值。被 cpu_reference.cpp 用 #include 实例化。",

    "include/cpu_reference.h",
    "CPU 基线三档的入口声明：kNaive（关闭向量化）、kScalarO3（标量优化）、"
    "kOpenMP（多线程）。每档编译选项独立、互不污染。",

    "include/cuda_utils.cuh",
    "CUDA 错误检查宏（CUDA_CHECK / CUDA_CHECK_LAST）、事件计时器（GpuTimer）、"
    "设备信息查询（QueryDevice / PrintDeviceInfo / QueryMemoryUsage）。"
    "当前包含 <cuda_runtime.h> 作为平台运行时入口。",

    "include/kernels.cuh",
    "GPU 力计算 kernel 声明：V3 档（rsqrtf + 打包 SoA + 软件流水展开）。"
    "LaunchKickDrift / LaunchComputeAcc（含自动 Pack）/ LaunchKick = 4 次启动 / 步。"
    "文件内注释解释了每一层优化的设计理由。",

    "include/diagnostics.h",
    "诊断量数据结构（DiagRecord：动能、势能、动量、角动量、max|a|、位力比）、"
    "DiagAccum 枚举（fp32/fp64 累加器）、GPU 工作区定义。",

    "include/diagnostics_gpu.cuh",
    "GPU 端诊断量计算声明：KernelDiagnostics<Acc>（O(N²) 的 FP64/FP32 诊断 kernel）、"
    "AllocateDiagnosticsGpu / FreeDiagnosticsGpu / ComputeDiagnosticsGpu。",

    "include/snapshot_stream.cuh",
    "轨迹快照 D2H 传输管线声明：pinned 双缓冲 + 独立 CUDA 流 + 异步 memcpy，"
    "将快照拷贝与下一步力计算重叠。",

    "include/sim_params.h",
    "参数解析与运行时配置：SimParams（步数/时间步长/软化/记录间隔/初值文件/输出路径）、"
    "CLI 解析（-n / -i / -s / -d / --block / --seed / --dim / --save / --diag-interval）、"
    "ic_generator 预设枚举。",

    "include/trajectory_io.h",
    "轨迹文件读写：二进制粒子主序格式（.bin）、JSON 诊断日志（.json），"
    "以及物理量定义（天体单位制 + G=39.4784176）。",

    # ---- src/：实现 ----------------------------------------------------------------
    "src/main.cu",
    "主入口：参数解析 → 初值生成（或文件加载）→ GPU 分配 → 主循环"
    "（踢-漂移 + 力计算 ×4 launch/步 + 快照异步回传会合）→ 计时 → 写盘。"
    "含 GPU vs CPU 交叉验证（CrossCheckAcc）与显存用量打印。",

    "src/kernels.cu",
    "GPU 力计算 kernel 实现：V3（rsqrtf + 打包 SoA + 内层循环手动展开 + "
    "条带化 tiling），本树唯一的力计算路径。每步由 LaunchComputeAcc 自动触发 "
    "LaunchPackPosMass（O(N) 拷贝，把标量 SoA 刷进 real4 缓存），积分器只更新 SoA。",

    "src/snapshot_stream.cu",
    "轨迹快照传输管线实现：Submit / Drain / 双缓冲轮转 / 同步点交会，"
    "保证快照回传不阻塞下一步力计算，且 main 线程在计时窗口外等待 I/O 完成。",

    "src/diagnostics_gpu.cu",
    "GPU 端诊断量实现：O(N²) 的动能/势能/动量/角动量/max|a| 计算。"
    "包含：块内共享内存归约 → 全局 atomicAdd 折叠、浮点诊断类型的位模式 atomicMax、"
    "AtomicMaxNonNegDouble（利用 IEEE 754 非负恒序性质）。",

    "src/cuda_utils.cu",
    "设备查询实现：cudaGetDeviceProperties → DeviceInfo 结构体 → PrintDeviceInfo。"
    "显存占用查询 cudaMemGetInfo。受 CUDA 驱动版本影响（部分云容器无此权限）。",

    "src/sim_params.cpp",
    "CLI 参数解析实现（getopt 风格）、SimParams 默认值填充、预设识别与名字映射。",

    "src/trajectory_io.cpp",
    "二进制粒子快照读写、JSON 诊断日志读写。严格遵循需求文档的粒子主序布局。",

    "src/diagnostics.cpp",
    "CPU 端诊断量计算：与 GPU 版同语义的 O(N²) 实现，用于 validate.sh 里"
    "GPU-vs-CPU 逐元素对比（段 7，即 GPU 诊断量的正确性判据）。",

    "src/cpu_reference.cpp",
    "CPU 基线三档的宿主实现：共享内核（cpu_kernel_inl.h）的三次 #include 实例化，"
    "分别以 kNaive / kScalarO3 / kOpenMP 开关控制。kOpenMP 档同时被 main.cu "
    "用作 GPU 加速度的交叉验证基准（CrossCheckAcc）。",

    "src/cpu_naive.cpp",
    "CPU 第 1 档：单线程标量、显式关闭向量化（-fno-tree-vectorize -fno-slp-vectorize）。"
    "独立成为 .cpp 是为了保证编译选项不对其他 TU 产生副作用。",

    "src/ic_generator.cpp",
    "初值生成器：Plummer 球、King 模型、均匀球、指数盘、双球碰撞、太阳系、"
    "二体圆/偏心轨道。编译为独立可执行文件 nbody_ic。",

    # ---- scripts/：构建与自动化 -----------------------------------------------
    "scripts/build.sh",
    "CMake 配置 + 编译：默认 sm_89，NBODY_ARCH 可切换。自动检测并清理异地拷贝的 "
    "CMakeCache.txt，打印编译工具链版本与 GPU 信息。支持 NBODY_FP64 / NBODY_FAST_MATH。",

    "scripts/bench.sh",
    "性能扫参脚本：N × BLOCK 二维网格，每组合 3 次预热 + 5 次计时取中位数。"
    "产出 results/bench/results.csv 与逐项 .json。并与同轮 CPU 基线配对计算加速比。",

    "scripts/validate.sh",
    "正确性门禁：8 段 16 项检查。段 1 二进制主序布局 / 段 2 初值生成 / 段 3 二体圆 / "
    "段 4 二体偏心 / 段 5 Euler vs Leapfrog / 段 6 多体 Plummer / 段 7 GPU 诊断量 / "
    "段 8 CSV 输出。任一 FAIL 则退出码非零。",

    "scripts/profile.sh",
    "nsys 与 ncu 采样：N=65536、20 步、block=128 的 nsys GPU trace + ncu 计数器。"
    "若 ncu 无权限（ERR_NVGPUCTRPERM，云容器常见），跳过 ncu 并提示补测方法。",

    "scripts/remote_all.sh",
    "远程一键脚本：build → validate → bench(全量扫参) → profile → "
    "打印可直接贴回文档的摘要 grep（能量误差、GFLOP/s 峰值、加速比）。",

    # ---- tools/：分析与可视化 ----------------------------------------------------
    "tools/visualize.py",
    "3D / 2D 轨迹动画生成器：读取 .bin 快照，matplotlib animation 输出 .mp4 或 .gif。"
    "支持 --trail（拖尾）、--frames（预览）、--dim {2,3} 模式。ffmpeg 缺失时自动降级为 GIF。"
    "碰撞挂起陷阱已诊断（82 万次 artist 更新 / 帧），运行时注意 N 过小导致每帧重绘时间超过 "
    "仿真时间。",

    "tools/check_collision_energy.py",
    "双球碰撞的初值能量分项对账脚本：读取 ic_generator 产出的文本初值文件，"
    "逐项算自能（W11/W22）、相互作用势能（W12）、轨道动能、内部动能，"
    "与生成器的解析预测（质点/Plummer/截断 Plummer）对账，裁定哪一公式更准。"
    "不画图，纯命令行输出。",

    "tools/plot_energy.py",
    "通用能量对比图生成器：接收多个 .log / .json 诊断文件，绘在同一张图里对比。"
    "validate.sh 段 5 用它将 Euler 和 Leapfrog 的 dE/E0 画在一起，1845× 差距一目了然。",

    "tools/plot_scaling.py",
    "性能 vs 规模的 roofline 图生成器：读 results.csv，绘 GFLOP/s–N 曲线，"
    "叠加理论峰值（--peak-tflops）与 ridge-point。用于说明 V3 在哪个规模转入计算受限。",

    # ---- video/：成品动画与图表 ---------------------------------------------------
    "video/cluster.mp4",
    "星团演化成品动画：Plummer 球 N=65536，3D + 拖尾 30（201 帧）。",

    "video/two_body.mp4",
    "轨道扰动成品动画：二体圆轨道 N=2，2D + 拖尾 60（889 帧）。",

    "video/two_body_euler.mp4",
    "Euler 对照动画：同初值下非辛积分器的轨道漂移。",

    "video/collision.mp4",
    "局部碰撞风险成品动画：双 Plummer 球对撞并合，3D + 拖尾 40（201 帧）。",

    "video/energy_comparison.png",
    "Leapfrog vs Euler 的 |dE/E0| 对比图（对数纵轴）。",

    "video/energy.png",
    "能量分项随时间变化图。",

    "video/scaling.png",
    "性能曲线三面板：每步耗时–N、TFLOP/s–N（带理论峰值线）、三档 CPU 加速比。",
]

def print_flat():
    """平面模式：每行 <路径> --- <说明>"""
    it = iter(INDEX)
    for path in it:
        desc = next(it)
        print(f"{path} --- {desc}")


def print_grouped():
    """按目录分组打印文件索引，每个条目一行 <路径> --- <说明>。"""

    groups = {
        "根目录": ["CMakeLists.txt", "README.md", "n_bodyversion.md", ".clang-format"],
        "data/":  ["data/" + f for f in [
            "params_bench.txt", "params_cluster.txt", "params_collision.txt",
            "params_solar.txt", "params_two_body.txt", "params_two_body_euler.txt"]],
        "include/": ["include/" + f for f in [
            "nbody_types.h", "cpu_kernel_inl.h", "cpu_reference.h",
            "cuda_utils.cuh", "kernels.cuh", "diagnostics.h",
            "diagnostics_gpu.cuh", "snapshot_stream.cuh",
            "sim_params.h", "trajectory_io.h"]],
        "src/": ["src/" + f for f in [
            "main.cu", "kernels.cu", "snapshot_stream.cu", "diagnostics_gpu.cu",
            "cuda_utils.cu", "sim_params.cpp", "trajectory_io.cpp",
            "diagnostics.cpp", "cpu_reference.cpp", "cpu_naive.cpp",
            "ic_generator.cpp"]],
        "scripts/": ["scripts/" + f for f in [
            "build.sh", "bench.sh", "validate.sh", "profile.sh", "remote_all.sh"]],
        "video/": ["video/" + f for f in [
            "cluster.mp4", "two_body.mp4", "two_body_euler.mp4",
            "collision.mp4", "energy_comparison.png", "energy.png",
            "scaling.png"]],
        "tools/": ["tools/" + f for f in [
            "visualize.py", "check_collision_energy.py", "plot_energy.py",
            "plot_scaling.py", "file_index.py"]],
    }

    # 扁平查找表：路径 → 说明（第 0 个元素 = 路径，第 1 个 = 说明）
    desc_map = {}
    it = iter(INDEX)
    for path in it:
        desc = next(it)
        desc_map[path] = desc

    for section_name, paths in groups.items():
        print(f"\n{'='*60}")
        print(f"  {section_name}")
        print(f"{'='*60}")
        for p in paths:
            d = desc_map.get(p, "（待补充）")
            print(f"  {p}")
            print(f"      {d}")

    # 未归属文件（自检用）
    covered = set()
    for paths in groups.values():
        covered.update(paths)
    missing = set(desc_map.keys()) - covered
    if missing:
        print(f"\n{'='*60}")
        print(f"  [未归属]")
        print(f"{'='*60}")
        for p in sorted(missing):
            print(f"  {p}")
            print(f"      {desc_map[p]}")


if __name__ == "__main__":
    import sys
    if "--group" in sys.argv or "-g" in sys.argv:
        print_grouped()
    else:
        print_flat()