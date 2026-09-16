# CUDA 金融衍生品定价器

这是一个使用 C++17 和 CUDA C++ 实现的蒙特卡洛定价命令行项目。它以 CPU 单线程实现作为基线，并提供 GPU 加速路径，用于定价欧式看涨期权和离散算术平均亚式看涨期权。

项目不是无参数可执行程序：运行 `pricer_cli` 时必须同时提供期权配置和模拟配置。下面的命令可从克隆仓库开始完成配置、构建、测试与一次实际定价。

## 功能范围

- 欧式看涨与离散算术平均亚式看涨期权；
- Black-Scholes 欧式看涨解析参考价；
- CPU 单线程蒙特卡洛基线；
- CUDA 蒙特卡洛路径，使用 cuRAND Philox 和 CUB 归约；
- FP32 / FP64 GPU 路径、FP64 统计量、标准误与 95% 置信区间；
- 每次运行输出 JSON 结果，并向性能 CSV 追加记录。

## 环境要求

- CMake 3.24 或更高版本；
- 支持 C++17 的主机编译器；
- CUDA Toolkit 12.x（含 cuRAND 与 CUB）及可用的 NVIDIA GPU；
- Ninja；
- 配置阶段能访问 GitHub：CMake 会获取固定版本的 nlohmann/json 3.11.3。

构建前需要按实际显卡指定 `CMAKE_CUDA_ARCHITECTURES`。例如，RTX 4060 Laptop 的计算能力为 8.9，对应 CMake 值 `89`。Linux/WSL 中可先查看：

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

去掉小数点后填入构建命令；例如 `8.6` 使用 `86`，`8.9` 使用 `89`。

## 构建

以下命令适用于 Linux 和 WSL。Windows 请在已初始化 MSVC 与 CUDA Toolkit 的开发者终端中执行同样的 CMake 命令；若使用 Ninja，Windows 可执行文件扩展名为 `.exe`。

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build -j
```

`89` 只是 RTX 4060 Laptop 的示例，不应照搬到其他显卡。正确性构建默认关闭 `PRICER_ENABLE_FAST_MATH`；性能测量也应使用 Release 构建。

## 运行定价器

先查看本机构建产物支持的参数：

```bash
./build/pricer_cli --help
```

运行欧式看涨期权，并同时执行 CPU 与 GPU 后端：

```bash
./build/pricer_cli \
  --option configs/option_european_call.ini \
  --simulation configs/simulation.ini \
  --output-dir results/european \
  --backend both \
  --repetitions 1
```

运行亚式看涨期权：

```bash
./build/pricer_cli \
  --option configs/option_asian_call.ini \
  --simulation configs/simulation.ini \
  --output-dir results/asian \
  --backend both \
  --repetitions 1
```

Windows + Ninja 时将可执行文件写为 `build\\pricer_cli.exe`，路径分隔符可使用 `\\`。若使用 Visual Studio 多配置生成器，则通常从 `build\\Release\\pricer_cli.exe` 启动。

`configs/simulation.ini` 默认配置为 1,000 万条路径、256 个时间步，适合作为完整实验输入，运行时间可能较长。需要快速调试时，请复制该文件到新的 INI，再降低 `num_paths` 与 `num_steps`；不要改写仓库提供的示例配置。

### 常用 CLI 参数

| 参数 | 说明 |
| --- | --- |
| `--option <path>` | 必填。欧式或亚式期权 INI 文件。 |
| `--simulation <path>` | 必填。路径数、步数、随机数、精度、block size 等模拟 INI 文件。 |
| `--output-dir <path>` | 输出目录，默认是 `results`。 |
| `--backend <cpu|gpu|both>` | 选择 CPU、GPU 或同时运行两者，默认 `both`。 |
| `--device <id>` | CUDA 设备编号，默认 `0`。 |
| `--warmup <n>` | GPU 非计量预热次数，默认 `1`。 |
| `--repetitions <n>` | 正式计量重复次数，默认 `1`。 |

## 配置与输出

`configs/` 中提供两类配置：

- `option_european_call.ini` 与 `option_asian_call.ini`：期权类型、现价、行权价、利率、波动率和到期时间；
- `simulation.ini`：FP32 GPU 默认实验配置；
- `simulation_fp64.ini`：FP64 GPU 配置，适合与 CPU FP64 基线进行同精度性能对比。

每个后端会在输出目录写入一份：

```text
<timestamp>-<option_type>-<backend>-result.json
```

性能数据会追加到同目录的 `performance.csv`。JSON 中价格、标准误、置信区间、运行时间和环境信息使用结构化字段记录；不可用统计值以 `null` 表示。

CPU 与 GPU 使用不同随机序列，验证时应比较统计一致性，而不是比较逐路径结果。性能结论应使用 CPU 与 GPU 的 `total_runtime_ms`，不要将 CPU 总时间与 GPU `compute_runtime_ms` 混用。

## 项目结构

```text
include/pricer/  公共 C++ 接口
src/             CPU、CUDA、CLI、统计与输出实现
configs/         示例期权与模拟 INI
results/         运行生成的 JSON、CSV 与实验产物（默认不提交）
```

## 常见问题

### CMake 提示未设置 CUDA 架构

本项目要求显式传入 `CMAKE_CUDA_ARCHITECTURES`。确认 GPU 的计算能力后重新配置，例如：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

不要复用由其他操作系统、其他 CUDA 版本或其他 GPU 生成的 CMake 缓存。

### FetchContent 下载依赖失败

检查 GitHub 网络与 Git 是否可用后重新配置。若所在环境已经有固定版本的依赖源码，也可以显式指定它们，避免 CMake 联网下载：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=/path/to/nlohmann_json \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest
```

这些目录必须分别包含对应依赖的顶层 `CMakeLists.txt`。不要将下载缓存、`_deps` 或构建目录提交到仓库。

### 无参数运行 `pricer_cli` 报错

这是预期行为。`--option` 与 `--simulation` 都是必填参数；请使用“运行定价器”章节中的完整命令。
