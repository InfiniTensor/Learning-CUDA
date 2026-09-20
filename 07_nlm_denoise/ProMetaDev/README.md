# GPU 非局部均值（NLM）实时图像降噪 —— 选题七（实时图像非局部均值降噪）

基于 CUDA 的非局部均值降噪程序：在尽量保留纹理与边缘的前提下抑制随机噪声，
可配置、可验证、可分析性能瓶颈，并给出**图像质量与处理延迟之间的权衡**。

- 支持**灰度（1 通道）与 RGB（3 通道）** 8 位图像，输入输出同格式（PNG / JPG / BMP 等）
- 权重严格按题目给定公式：`w(p,q) = exp(-max(dist(P_p,P_q) - 2σ², 0) / h²)`
- `patch_radius` 支持 1~5（题目要求 ≥3），`search_radius` 支持任意值（题目要求 ≥10）
- 提供**精确 CPU 参考实现**与 **OpenCV `fastNlMeansDenoising*`** 两套对照
- 提供三档可选的近似加速开关（搜索抽稀 / patch 抽稀 / 权重查表），并量化其误差

---

## 目录结构

```
proj_nlm/
├── include/
│   ├── types.h        # 图像、NLM 参数、性能统计、质量指标
│   ├── image_io.h     # 图像读写（OpenCV）+ 质量指标
│   └── nlm.h          # CPU 参考 / OpenCV 参考 / GPU 接口 + 语义定义
├── src/
│   ├── main.cpp       # CLI: denoise / bench / selftest
│   ├── image_io.cpp   # 参数解析与校验、图像读写、MAE/PSNR
│   ├── cpu_ref.cpp    # CPU 精确 NLM（OpenMP）+ OpenCV 参考封装
│   └── nlm_gpu.cu     # GPU NLM：按位移分解 + 可分离盒式求和 + 共享内存分块
├── tools/
│   └── gen_image.py   # 合成测试图（干净图 + 加噪图）
├── tests/
│   ├── run_tests.sh   # 端到端测试
│   └── benchmark.sh   # 分辨率/参数/近似开关基准
├── params/            # 示例参数文件
├── CMakeLists.txt
├── build.sh
└── README.md
```

## 依赖

| 组件 | 用途 | 是否必需 |
|---|---|---|
| CUDA Toolkit ≥ 11.0（实测 12.2） | GPU 计算 | 必需 |
| C++17 编译器（实测 g++ 9.4） | 主机端 | 必需 |
| **OpenCV 4**（`libopencv-dev`） | 图像读写（PNG/JPG）+ 题目要求的 `fastNlMeansDenoising*` 对照 | 必需 |
| OpenMP | CPU 参考实现并行 | 可选 |

> 核心算法只依赖 CUDA；OpenCV 仅用于图像编解码与外部对照，替换为其它解码器不影响算法本身。

## 构建

```bash
bash build.sh [SM_ARCH]     # 默认 90；同时生成 PTX 以兼容更新架构（如 sm_120）
```

产物：`build/nlm`

## 使用

```bash
# 自检
./build/nlm selftest

# 降噪（写出结果图）
./build/nlm denoise <in.png> <params.txt> <out.png> [选项]

# 只跑一遍并打印指标（不强制写图）
./build/nlm bench <in.png> <params.txt> [选项]
```

### 选项

| 选项 | 说明 |
|---|---|
| `--cpu` | 额外运行 CPU 精确参考实现并对比（1080p 以上会较慢） |
| `--opencv` | 额外运行 OpenCV `fastNlMeansDenoising*` 作为外部对照 |
| `--ref <clean.png>` | 提供**无噪真值图**，计算降噪后的 PSNR/MAE（最有意义的质量指标） |
| `--compare <img>` | 与任意另一张图逐像素比较 |
| `--perf <log>` | 性能日志输出路径（追加写） |

### 参数文件（与题目定义一致）

```ini
patch_radius  = 3      # patch 半径，实际 patch 边长 (2*rp+1)
search_radius = 10     # 搜索窗口半径
h             = 10.0   # 滤波强度
sigma         = 25.0   # 噪声标准差估计（8 位图像，0-255）

# ---- 可选：近似加速开关（默认关闭，即精确算法）----
search_step = 1        # 搜索窗口采样步长，>1 为近似
patch_step  = 1        # patch 采样步长，>1 为近似
use_lut     = false    # 用查找表近似 exp()
lut_bins    = 4096     # 查找表分档数
```

## 算法设计

### 规范语义（本项目精确定义，CPU/GPU 一致）

记 `p` 为中心像素、`q` 为搜索窗口内候选、`P` 为以 `p` 为中心的 patch：

```
D(p,q) = Σ_c Σ_{k∈P} ( I_c(clamp(p+k)) - I_c(clamp(q+k)) )²     逐通道求和的平方差
w(p,q) = exp( -max( D(p,q)/|P| - 2σ², 0 ) / h² )                 |P| 含通道数
out(p) = Σ_q w(p,q)·I(q) / Σ_q w(p,q)
```

- 候选 `q` 的搜索范围**裁剪到图像内**（边界像素候选数自然减少）
- patch 读取在边界处按**复制边缘（clamp）**处理，保证 patch 始终完整
- `D` 除以 `|P| = nk²·C` 得到逐像素逐通道的平均平方差，故 `2σ²` 恰好是"纯噪声下该量的期望"

> 归一化里的 **通道数 C 不能漏**：`D` 累加了 `nk²·C` 个平方差，只除以 `nk²` 会把彩色图的距离放大 3 倍，
> 使 `dd ≈ (C-1)·2σ²` 远大于 0 → 除中心外所有候选权重趋近 0 → 输出≈输入（开发中真实踩过的坑）。

### 关键优化：按位移分解 + 可分离盒式求和

直接做法是"每个像素遍历搜索窗口、每次重算整块 patch 距离"，计算量 `O(S²·P²)`
（`S=2rs+1`、`P=(2rp+1)²`），在 1080p 上是 `441 × 49 × 3 ≈ 6.5×10⁴` 次乘加/像素——GPU 上也要上百毫秒。

本实现把它改写成**数学等价**的两步形式，每个位移只需一遍可分离累加：

```
对每个位移 s=(dx,dy)：
  G_s(u)   = Σ_c ( I_c(clamp(u)) - I_c(clamp(u+s)) )²      // 每个像素 1 次，与 patch 无关
  D(p,p+s) = Σ_{k∈P} G_s(p+k)                              // P×P 盒式求和 → 可分离为 横向+纵向 两次累加
  w(p,s)   = exp( -max( D/|P| - 2σ², 0 ) / h² )
  num += w·I(clamp(p+s));  den += w
```

计算量降到 `O(S²·(2·nk + 1))`，1080p 上约 `441 × 15 ≈ 6.6×10³` 次/像素——**降低约一个数量级**，
且每一步都是规则访存，非常适合 GPU。该改写**不改变任何数学结果**（GPU 与 CPU 参考 MAE=0.0000）。

### GPU 实现要点

| 点 | 做法 |
|---|---|
| **并行粒度** | 一个线程 = 一个输出像素（含全部通道）；`block = 32×8 = 256` 线程 |
| **共享内存分块** | 块内共享 `(32+2rp) × (8+2rp)` 的 `G_s` 缓冲（rp=3 时 3.9 KB），含 halo，**只从全局读一次** |
| **边界** | 全局索引 `clamp` 到图像内（patch 复制边缘）；候选越界则跳过该位移（搜索窗口裁剪） |
| **位移循环** | 位移循环放在**内核内部**，`num/den` 常驻寄存器，避免每个位移都写回全局内存 |
| **通道** | 逐通道平方差先求和再算权重 → **精确彩色 NLM**（非通道独立近似） |
| **权重** | 默认 `__expf`（硬件加速指数）；可选按**指数 `d/h²`** 分档的查找表 |
| **模板实例化** | `patch_radius ∈ {1..5} × 通道数 ∈ {1,3}` 共 10 个实例，内层循环可展开 |

## 近似开关（质量-速度权衡）

| 开关 | 含义 | 性质 |
|---|---|---|
| `search_step = k` | 搜索窗口每隔 k 个候选取一个 | **有损**：候选数降至 1/k²，质量随之下降 |
| `patch_step = k` | patch 每隔 k 个采样点取一个 | **有损**：patch 距离估计方差变大 |
| `use_lut` + `lut_bins` | 用查找表近似 `exp()` | **近无损**：按指数分档，2048 档时与精确实现 MAE≈0.06 |

> 查找表必须按**指数 `x = d/h²`** 分档，而不是按距离 `d` 分档：
> `h` 较小时 `d` 的档宽会被 `h²` 放大成很大的指数跨度，权重严重失真（实测按 `d` 分档 MAE 达 3.7）。

## 性能

> **重要说明（笔记本 GPU 热降频）**：本机 RTX 5070 Ti Laptop 在持续负载下会明显降频
> （实测 SM 时钟 **2257~2385 MHz / 上限 3090 MHz**、温度 86~87 °C）。同一负载在冷态与热稳态下
> 相差可达 1.6 倍以上。因此下表同时给出**冷态**（GPU 空闲降温后）与**热稳态**（连续压测）两组
> 数据，并标注 GPU 时钟；`tests/benchmark.sh` 每次运行前后都会打印时钟/温度，且**每档重复 3 次取中位数**。

`rp=3, rs=10, h=10, sigma=25`，RGB，单位 ms：

| 分辨率 | 冷态总 ms | 冷态内核 ms | 冷态加速比(vs OpenCV) | 热稳态总 ms | 热稳态加速比 |
|---|---|---|---|---|---|
| 480×270 | 3.02 | 2.14 | 93.3× | 5.99 | 56.9× |
| 1280×720 | 18.48 | 12.38 | 18.0× | 24.68 | 17.7× |
| **1920×1080** | **43.51** | 28.49 | **10.1×** | **70.50** | **7.6×** |
| **3840×2160** | **175.64** | 120.71 | **7.4×** | **283.69** | **5.5×** |

冷态吞吐约 47~50 MPix/s；相对 **CPU 精确参考（OpenMP 32 线程）加速 160×（480×270）
~ 189×（1280×720）**，且两者对无噪真值的 PSNR 完全相同。完整分析见 `report.md` 第 5 节。

## 正确性验证

| 验证项 | 结果 |
|---|---|
| 引擎自检（合成双段图，GPU vs CPU 参考） | PASS（MAE=0.0000） |
| **GPU vs CPU 精确参考**（灰度 256²、彩色 384²） | **MAE = 0.0000（最大差 1/255，仅浮点舍入）** |
| 降噪有效性（相对无噪真值） | PSNR 20.19 dB（含噪）→ 33.70 dB（降噪后） |
| 与 OpenCV `fastNlMeansDenoisingColored` 的差异 | MAE≈2.4（两者 patch 权重与边界策略不同，见报告） |
| 近似开关误差 | LUT 近无损（MAE 0.06）；采样类近似质量损失已量化 |
| 参数校验（非法 patch_radius / h / 缺文件） | 均明确报错并返回非零退出码 |
| 结果确定性（重复运行） | 输出逐字节一致 |

## 局限

1. **未做国产平台适配**（题目提到每适配一款可加分）。
2. **未采集 `ncu` / `nsys` 数据**（题目列为加分项）：本机为 WSL2，不透传 CUPTI
   （此前项目实测 `nsys` 的 CUDA kernel 段为空）；且 GPU 为 sm_120，CUDA 12.2 只能 JIT 运行 sm_90 PTX。
3. **`search_radius` 很大时耗时线性增长**：位移数是 `O(rs²)`，`rs` 从 10 增到 15 耗时约翻倍。
   可用"按位移并行"（一次启动处理多个位移）进一步压榨，见报告的未来方向。
4. **未实现多尺度 / 金字塔 NLM** 等更高级的加速结构。
5. OpenCV 对照只在精确参数下提供（其 `fastNlMeansDenoising*` 不接收 `sigma`，也不支持抽稀参数）。

## 复现

```bash
bash build.sh 90
./build/nlm selftest
bash tests/run_tests.sh 90
bash tests/benchmark.sh 90
REP=5 bash tests/benchmark.sh 90          # 提高重复次数（默认 3 次取中位数）
ONLY=1 bash tests/benchmark.sh 90         # 只跑分辨率扫描（冷态测量）

# 生成 1080p 测试图并降噪（建议放在 Linux 原生路径，/mnt/c 走 Windows 挂载会慢一个数量级）
python3 tools/gen_image.py --size 1920x1080 --channels 3 --sigma 25 --seed 5 \
    --out-clean /tmp/clean.png --out-noisy /tmp/noisy.png
./build/nlm denoise /tmp/noisy.png params/exact.txt /tmp/out.png \
    --ref /tmp/clean.png --opencv --perf /tmp/nlm.log
```
