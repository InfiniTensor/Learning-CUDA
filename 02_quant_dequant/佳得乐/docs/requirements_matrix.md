# 要求映射与验收

依据：项目 PDF 总则与选题二，以及补充要求。
实现位于根目录。下表证据相对本目录；最终运行证据为最新一次 `results/<run_id>/`，汇总由原始 CSV 自动生成。只有该目录 `SUCCESS` 与测试日志均存在且通过才算运行项通过。

|编号/来源|必做要求|实现位置|验证/证据|状态|
|---|---|---|---|---|
|1 格式|MXFP8-E4M3|numeric.hpp; configs/e4m3.toml|scalar_tests; integration_gpu; results/*e4m3*|已实现，实测通过|
|2 格式|MXFP8-E5M2|numeric.hpp; configs/e5m2.toml|同上 e5m2|已实现，实测通过|
|3 格式|NVFP4-E2M1|numeric.hpp; configs/nvfp4.toml|同上 nvfp4|已实现，实测通过|
|4 文件|FP32/FP16文件→低精度文件→独立CUDA反量化|src/main.cpp, io.cpp, gpu.cu|results/*quantize/dequantize/verify.log|已实现，实测通过|
|5 顺序|数值→CPU→GPU→优化顺序|docs/numerics.md;各阶段源码|同一次 run 内 baseline/optimized 对照；report.md|通过|
|6 环境|CPU/系统/编译器/CMake/GPU/toolkit工具检测|scripts/reproduce.py|environment.log, CMakeCache.txt|通过|
|7 环境|真实CUDA探测|tests/probe.cu|probe.log，设备枚举/分配/kernel/同步/结果|通过|
|8 环境|sm_86且可配置|CMakeLists.txt CUDA_ARCH|configure/build.log；metadata.json|3090通过，其他硬件未测|
|9 冲突处理|保留独立测试、清理生产临时调试|tests/与src/分离|README解释|通过|
|10 数值|一手资料/版本/数值规则|docs/numerics.md|OCP v1.0、NVIDIA 2025-06-24链接|已查阅|
|11 数值|三种位布局、bias、正规/次正规、极值、特殊值|numeric.hpp::decode/encode|全256/256/16枚举与独立oracle|通过|
|12 数值|E4M3具体变体|numerics.md|OCP finite ±0，非FNUZ|通过|
|13 数值|E8M0真实编码/恢复|mxscale, scale_value|全255 scale编码与边界|通过|
|14 数值|NV局部E4M3+全局FP32及方向|global_scale, nvscale|完整配置、存储重载/逐位比较|通过|
|15 数值|用编码后scale量化|cpu.cpp/gpu.cu|CPU/GPU scale/packed/output逐位|通过|
|16 数值|FP32/FP16输入×三输出×两scale×两round|Config;cpu/gpu|72配置×8shape×3分布=1728主用例|通过|
|17 数值|RNE及不同舍入阶段区分|numeric.hpp; numerics.md|中点/两侧，FP16/BF16 tie测试|通过|
|18 数值|严格距离概率随机、seed/index确定性|uniform,encode|100000样本概率检验、两GPU variant逐位|通过|
|19 数值|±0/全零/极值/离群/scale范围/NaNInf|numeric.hpp; integration.cpp|结构化块、FLT_MAX、最小次正规；非有限拒绝|通过|
|20 数值|行方向分块/尾块/tensor扩展元数据|io.cpp;file_format.md|3×17和块边界形状；extension标志验证|通过|
|21 数值|非默认块大小标记或拒绝|validate|非法17拒绝|通过，仅支持默认值|
|22 CPU|独立CPU C++17、不依赖CUDA|ENABLE_CUDA=OFF|build-cpu/CTest；cpu-release.log|通过|
|23 CPU|独立解码与距离参考|tests/scalar.cpp::oracle|指数位权求和；穷举最近邻|通过|
|24 CPU|CPU优化编译/线程数明确|CMakeLists; metadata|Release -O3，单线程|通过|
|25 配置|指定配置字段、非法/不适用拒绝|read_config|配置负例；NV fp8_type拒绝|通过|
|26 配置|magic/version/endian/字段宽度/shape/偏移|io.cpp;file_format.md|容器往返，损坏元数据拒绝|通过|
|27 文件|真实4bit打包/奇数padding/无行padding|quant_cpu,encode_kernel|手算0x32/0x07；奇数51元素26字节|通过|
|28 文件|独立dequantize仅依赖文件|main.cpp|独立进程命令日志|通过|
|29 文件|非法shape/乘法溢出/dtype/format/截断/越界|checked_size,load_packed|逐长度截断；各元数据字段破坏|通过|
|30 CUDA|GPU归约/scale编码/量化/打包/输出转换|gpu.cu全部kernels|RTX3090逐位对照|通过|
|31 CUDA|合理baseline并行度|reduce_all,make_scales|256线程归约，每warp一个量化块|通过|
|32 CUDA|NV byte单线程写、packed load|encode/decode_kernel|racecheck 0 hazards|通过|
|33 CUDA|BF16软件输出转换|bf_bits|全部BF16往返、tie、全输出配置|通过|
|34 CUDA|API/launch/同步检查和RAII|check,Device,Events|sanitizer 0 errors；异常路径RAII|通过|
|35 测试|全编码、相邻中点、指数边界、特殊值|scalar.cpp|scalar_tests PASS|通过|
|36 测试|scale零/常数/混合/离群/编码边界|scalar/integration|结构化块288例、scale边界|通过|
|37 测试|文件关闭重开/尾部/非法输入|io.cpp,integration|CPU和GPU独立加载|通过|
|38 测试|均匀/正态/离群固定seed|generate; quality.cpp|metadata.json; quality.csv 216行|通过|
|39 测试|packed/scale/输出分别逐位对照|equal重载|1728 CPU/GPU组合，不放宽容差|通过|
|40 测试|实现误差与重建误差分开|equal vs error|位差为0；quality.csv/性能CSV误差|通过|
|41 测试|FP16实际存储值参考、累加精度|error uses stored input,long double|质量表分输入/输出dtype|通过|
|42 测试|compute-sanitizer代表用例|full脚本|memcheck/racecheck.log|0 errors/0 hazards|
|43 性能|CPU/baseline/optimized保留|CLI/CMake|三后端/variant命令|通过|
|44 性能|小/中/大矩阵|reproduce.py|32²/512²/2048²；显存24GiB|通过|
|45 性能|Events、warmup/repeat、波动|gpu.cu;main.cpp;summarize.py|warmup3/repeat5，每轮CSV，均值/std|通过|
|46 性能|scale/encode/fullquant/dequant/H2D/D2H|Timing和同轮event边界|CSV分项；fullquant包含归约和初始化|通过|
|47 性能|含传输流程/含文件I/O分别计时|main.cpp外层steady_clock|process_ms/e2e_ms/cpu_e2e_ms|通过，见范围说明|
|48 性能|公平对应阶段比较|main.cpp|CPU同输入同配置；含传输完整API；文件双方同流程|通过|
|49 性能|max/MAE/MSE/压缩率/时间/带宽/加速比|benchmark/quality|原始CSV及summary|通过|
|50 性能|payload+全部scale、完整文件压缩率|main.cpp|两输入dtype分别；NV计全局4字节|通过|
|51 性能|带宽流量/公式明确|report.md|逻辑最小反量化读写，非DRAM测量|通过|
|52 性能|总体性能超越CPU|report.md实测表|中/大规模同流程；小矩阵失败如实记录|有条件达到|
|53 优化|先baseline证据再优化、前后同条件|同一次 run 内 baseline/optimized CSV|位域解码单项优化，未凑数量|通过|
|54 优化|ncu/nsys分析或明确权限阻塞|run 内 nsys/nsys-stats.log；ncu 失败记录|ncu ERR_NVGPUCTRPERM；nsys 时间线成功|ncu受限，已记录|
|55 工程|目录结构与五个CLI|README;src/main.cpp|实际命令执行|通过|
|56 工程|一键quick/full、非零失败、run_id产物|scripts/reproduce.py|quick/full SUCCESS|通过|
|57 工程|源码/生成物分离、限定清理|.gitignore;scripts/clean.py|只清理指定且带metadata的run|已实现|
|58 文档|报告分节、真实CSV图表|docs/report.md;summarize.py|summary.md/json/csv，PNG/SVG|通过|
|59 文档|按步骤学习指南、中文原因注释|docs/learning_guide.md;核心函数|文件/函数/输入输出/手算和命令|通过|

## 明确边界

未做 OCP 全操作一致性认证：使用其数值编码和默认块参数，矩阵元素转换固定 SAT；未提供单独 OVF 转换模式。tensor/尾块为项目扩展。没有原生 Tensor Core 路径、其他 GPU 或国产平台实测。ncu 性能计数器缺权限不影响普通 CUDA 测试，但不能声称完成 DRAM 流量分析。
