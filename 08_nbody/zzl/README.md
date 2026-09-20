# N 体引力模拟与可视化

本项目计算粒子间的两两引力作用，并用 CUDA 并行更新粒子状态。程序提供 CPU/OpenMP 对照实现，支持 Euler 和 Leapfrog 积分；轨迹保存为二进制文件，由 Python 绘制二维或三维动画。
实现与实测分析见 [实验报告](REPORT.md)。

## 环境与构建

需要 C++17 编译器、CMake 3.24+、NVIDIA CUDA Toolkit、支持 CUDA 的显卡和 Python 3.11+。
归档计算程序的验收环境为 WSL Ubuntu 24.04、CUDA 12.8、RTX 5060 Laptop。
Windows 可使用 Visual Studio C++ 与 CUDA 从源码构建；包内预编译程序面向 WSL/Linux。

在项目根目录运行：

~~~bash
python -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j 4
./build/nbody --device-info
~~~

如需指定 nvcc 路径，配置时追加 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc。
CPU 版本配置时使用 -DNBODY_ENABLE_CUDA=OFF，运行时显式指定 --backend cpu；CPU 测试使用 tests/test_nbody.py 并省略 --require-cuda。换机器时重新配置构建目录。
包内 bin/nbody 是本次 Linux 构建，解压后可 chmod +x bin/nbody；其他 GPU 建议从源码构建。

## 运行示例

~~~bash
./build/nbody --input examples/binary.txt --config examples/binary.cfg --output outputs/run/trajectory.bin --log outputs/run/performance.json --final-state outputs/run/state.csv
python visualize.py outputs/run/trajectory.bin --dimension 2 --save outputs/run/orbit.gif
~~~

每次使用新的输出路径，程序拒绝覆盖已有结果。服务器环境可设置 MPLBACKEND=Agg 导出动画。
--dimension 3 绘制三维动画；--reference 对比扰动前后轨迹；--risk-distance 设置近距离阈值。
--max-pairs 限制单帧配对数量，超限记录 truncated；近距离筛查针对保存帧中的粒子间距。

## 输入输出

粒子文本每行 x y z vx vy vz mass；质量非负且总质量为正。配置示例：

~~~text
dt=0.001
num_steps=1000
record_interval=10
G=1
softening=0.02
integrator=leapfrog
~~~

支持 BOM、注释和引号。dt、softening、record_interval 必须为正，G 非负。
二进制轨迹为小端 int32 P、int32 R，随后为 float32[P][R][3]，长度 8+12PR 字节；
附带 JSON 记录真实 steps 和 dt，始终包括初态与末态。Python 用 NumPy memmap 读取。

| 参数 | 用途 |
|---|---|
| --backend cpu/cuda | 后端，默认 cuda |
| --kernel naive/tiled | 求力内核，默认 tiled |
| --block-size 128/256/512 | CUDA 线程块，默认 256 |
| --cuda-execution auto/direct/graph | CUDA 提交模式 |
| --cpu-threads N | CPU 线程数 |
| --no-trajectory | 关闭轨迹 |
| --diagnostics auto/full/momentum | 守恒量诊断 |
| --compare-cpu | 加跑相同步数的 CPU 对照 |
| --final-state PATH | 保存最终状态 CSV |

性能日志区分 GPU Event 计算时间、主机计算区间、回传、轨迹追加、转置收尾、最终状态输出及进程总时间。
完整 CPU 对照会计入总耗时，独立 GPU 运行时间应通过单独执行 GPU 作业测量。
转置目标块为 32 MiB，另有行缓冲和单帧数组；单粒子历史可超过预算。磁盘约需两份轨迹空间。

## 验证与复现

~~~bash
python tests/test_nbody.py --exe build/nbody --require-cuda --report outputs/tests.json
python tests/test_trajectory_reader.py
python tests/test_package.py
python tests/test_verify_nsys.py
python scripts/benchmark.py --exe build/nbody --out outputs/benchmark-new --evidence outputs/benchmark-new.json
python scripts/benchmark_end_to_end.py --exe build/nbody --data outputs/benchmark-new --out outputs/end-to-end-new --evidence outputs/end-to-end-new.json
python scripts/make_demos.py --exe build/nbody --out outputs/demos-new
~~~

归档 CUDA 验收结果在 evidence/submission-tests.json，性能结果在 evidence/submission-benchmark.json 和
evidence/submission-end-to-end.json。测试代码位于独立的 tests/ 目录。
outputs/demos/ 是示例动画与数据；分析器历史日志用于记录工具与平台的兼容情况。
tests/ 中 NumPy 参考、数值收敛、守恒量、尾块、轨迹格式及 GIF 渲染均独立检查。

本机于 2026-09-20 从同一源码树在独立构建目录重新配置并编译，复跑同一验收，26 项结果与归档记录一致；
构建输出见 evidence/submission-clean-build.txt，复跑记录见 evidence/submission-audit-current.json 与
evidence/submission-audit-current.txt。该独立构建的程序哈希与归档程序不同，因此只用于核对源码可复现性，
不替代归档程序的验收记录。

轨迹读取器使用内存映射，并按最多 1 MiB 的数据块检查非有限数值；
即使只有一个粒子、记录时间很长，单次检查也保持这个上限。
该上限适用于检查用临时数组；总进程内存还包括程序状态和文件映射等开销。
JSON 侧文件若存在，必须声明 float32，粒子数和记录数必须为整数；步数从零开始严格递增，
范围为 0 到 2147483647，时间步长必须为正有限数值。布尔值按非法类型处理。
缺少侧文件时按记录序号读取，时间轴退化为记录序号。
元数据类型与步数溢出用例见 tests/test_trajectory_reader.py；历史 CUDA 回归
由项目开发记录保存；交付包提供当前测试入口。
已有交付包与 submission-* 证据仍对应各自封存版本。

打包前会核对验收清单中的源码及程序 SHA256、验收记录与基准所对应的程序版本，
并要求全部 CUDA 验收用例执行通过。计算源码变更后重新构建并执行 CUDA 验收；工具脚本变更后执行相关回归。
记录验证结果后更新 evidence/submission-artifact.json 中的 source_sha256，再生成新包。
该清单需包含 CMakeLists.txt、三个顶层 Python 模块以及 src/、include/、tests/、scripts/ 下的源码。
验收计数采用严格整数类型检查。
每个压缩包固定保存生成时的版本，后续修订通过新包交付。

## 目录

- src/、include/：C++ 与 CUDA 源码。
- generate.py：生成二体、星团、轨道扰动和相遇数据。
- trajectory.py：二进制读回与空间哈希近距离筛查。
- visualize.py：交互动画和导出。
- scripts/：基准、演示、报告和打包工具。
- evidence/：最终验收、性能和分析器记录。
- REPORT.md：方法、结果、误差与适用范围。

本项目的计算范围为 CPU/OpenMP 与 NVIDIA CUDA，状态采用单精度表示，求力采用直接算法，轨迹通过同步回传保存。
交付包包含源码、测试、报告、演示与本次 Linux 程序。

性能分析的实验条件、归档结果和复现命令见 [性能分析说明](docs/PROFILING.md)。

当前打包流程为每个压缩包生成同名 `.zip.sha256`，并生成记录压缩包 CRC 与包内文件哈希校验结果的 `.zip.verification.json`；校验记录通过压缩包 SHA-256 绑定具体版本。程序运行测试以独立验收记录为准。
