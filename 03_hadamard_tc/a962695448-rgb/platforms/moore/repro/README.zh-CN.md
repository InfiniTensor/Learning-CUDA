# 摩尔 S4000：公开复现材料

本目录公开 2026-09-16 已完成实验的原始计时、验证清单、冻结对照源码和复算工具。
整理阶段只执行本地 CPU 检查，没有重新开启 S4000 或生成新的 GPU 性能结果。

## 文件与口径

| 文件 | 用途 |
|---|---|
| recorded/paired-round1.csv 至 paired-round3.csv | 三个独立进程的 6,750 条原始计时 |
| recorded/optimization.json | 原始实机阶段、返回码、命令、输入和输出哈希 |
| recorded/full-run-summary.json、full-validation.json | 1,504 组完整输入、192 项 API、14 项 CLI 检查 |
| recorded/disabled-run-summary.json、disabled-validation.json | 关闭 Shuffle32 的兼容性结果 |
| recorded/quant-gpu.log | 2,074,568 项 GPU 量化对照，三方零差异 |
| recorded/cpu-equivalence-*.json | 原始 macOS/Linux CPU 检查，包含夹具 SHA-256 |
| control/control_api.h、control_api.mu | 冻结旧版，仅调整命名空间与头文件名以同程序链接 |
| paired_benchmark.mu、check_quant_gpu.mu、check_quant_cpu.cpp | 与实测版本字节一致的计时及对照程序 |
| check_quant_cpu.py | 可指定库文件、输出目录的 CPU 夹具生成器 |
| analyze_optimization.py | 离线核验 CSV 完整性、哈希、单位及全部配对结果 |
| MANIFEST.json | 公布文件的来源、哈希及整理阶段的检查范围 |

原始记录中的 /data 路径属于当时服务器的实验现场，用于追溯。下面的复现命令使用仓库相对路径。
账户连接信息、关机截图、构建二进制和 25 MB 夹具不在此目录中；夹具可从源码精确再生成。

## 1. 离线重算原始结果

从 03_hadamard_tc/a962695448-rgb 项目目录运行，只需要 Python 标准库：

~~~bash
moore_work=$(mktemp -d)
python3 platforms/moore/repro/analyze_optimization.py --output "$moore_work/analysis"
python3 -m unittest discover -s platforms/moore/repro -p 'test_*.py' -v
~~~

输出目录必须尚不存在。analysis.json 保留全部正负例，paired-comparisons.csv 给出各轮、
各配置旧版/新版的耗时比。脚本检查每轮 2,250 条、每个配置/方法组合的五个分组、
固定的 30 个形状/精度配置、15 个方法/操作组合以及微秒和毫秒换算；
篡改、缺行、重复或未知配置都会失败。

默认 Optimized 分步与融合在 30 个配置的每轮都减少至少 5% 耗时，原始验收通过。
融合旧版/新版比值为 2.012～2.898×；它是同设备、热输入、MUSA event 区间的比较，
不代表应用端到端收益。完整解释见 [实测报告](../REPORT.zh-CN.md)。

## 2. 在 CPU 上重新生成测试夹具

需要支持 C++17 的编译器及 NumPy。原始 macOS/Linux 分别使用 NumPy 2.3.5/1.23.5，
两者生成了相同字节。沿用上节新建的 moore_work：

~~~bash
c++ -std=c++17 -O2 -fno-fast-math -ffp-contract=off -shared -fPIC \
  -Iplatforms/moore -Iinclude platforms/moore/repro/check_quant_cpu.cpp \
  -o "$moore_work/libcheck_quant.so"
python3 platforms/moore/repro/check_quant_cpu.py \
  --library "$moore_work/libcheck_quant.so" --output "$moore_work/cpu"
~~~

该动态库使用既有 CPU 除法/舍入参考检查实际整数实现，共 17,527,208 项；
随后生成 2,074,568 条小端记录，每条为 float32 value、float32 scale、int32 expected。
GPU 夹具与 CPU 检查重叠，不将两类数量相加当作独立输入。

预期夹具大小为 24,894,816 字节，SHA-256：

~~~text
b404de04128361ed75347666113cfe2545f7c6e52aa828b070c1812c967ce230
~~~

原服务器参考头文件比 GitHub 中的共享参考多一个末尾空行。计算代码一致；公开版本已在
CPU 上重建，并检查生成夹具的字节哈希与原实机输入完全相同。

## 3. 在另行准备的 MUSA 设备上重建

以下命令需要实际 MUSA SDK 和设备；不属于离线复算步骤。原实测环境为 S4000、
MUSA 5.1.0、mp_22。CPU 夹具先按上一节生成。

~~~bash
musa_root=/usr/local/musa
"$musa_root/bin/mcc" -std=c++17 -O2 -fno-fast-math -ffp-contract=off \
  --offload-arch=mp_22 -Iplatforms/moore \
  platforms/moore/repro/check_quant_gpu.mu \
  -L"$musa_root/lib" -lmusart -o "$moore_work/quant_gpu"
LD_LIBRARY_PATH="$musa_root/lib:${LD_LIBRARY_PATH:-}" \
  "$moore_work/quant_gpu" "$moore_work/cpu/quant_fixture.bin"

"$musa_root/bin/mcc" -std=c++17 -O2 -fno-fast-math -ffp-contract=off \
  --offload-arch=mp_22 -DHADAMARD_MOORE_SHUFFLE32 \
  -Iplatforms/moore -Iplatforms/moore/repro/control -Iinclude \
  platforms/moore/hadamard_api.mu platforms/moore/repro/control/control_api.mu \
  platforms/moore/repro/paired_benchmark.mu \
  -L"$musa_root/lib" -lmusart -o "$moore_work/paired_benchmark"

for round in 1 2 3; do
  LD_LIBRARY_PATH="$musa_root/lib:${LD_LIBRARY_PATH:-}" \
    "$moore_work/paired_benchmark" --benchmark --groups 5 --repeats 100 \
    --csv "$moore_work/paired-round${round}.csv" \
    > "$moore_work/paired-round${round}.log" 2>&1
done

python3 platforms/moore/run_platform.py --shuffle32 --no-benchmark \
  --output "$moore_work/full"
python3 platforms/moore/run_platform.py --quick --no-benchmark \
  --output "$moore_work/disabled"
~~~

新运行会产生自己的 CSV 和日志，不能覆盖 recorded 目录或套用原始哈希清单。
analyze_optimization.py 默认针对已归档的实验；分析新数据时需提供相应的
optimization.json，至少记录通过状态、三份 CSV 的实际 SHA-256，并另外保存
环境、编译参数、测试与源码来源。原始清单提供完整格式示例。

## 整理阶段的验证范围

本次公开整理保留 GPU 计算源码和冻结对照的原字节；调整的是 CPU 生成器的路径参数、
离线复算入口和文档。原始三轮数据可以重新得到与此前完全相同的分析结果。
GPU 源码已有实机证据，本轮没有再编译或运行 MUSA 程序。
