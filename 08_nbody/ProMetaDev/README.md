# CUDA N-Body Gravity Simulation (2026 Summer)

一个基于 CUDA 直接求和 O(N²) 的 N 体引力仿真项目，专为 GPU 加速学习 / 2026-summer 课程项目设计。
特性覆盖：**两种加速度 Kernel（simple / shared_tiling）+ 两种积分器（Euler / Leapfrog）+ SoA 内存布局 + 严格二进制轨迹格式 + CSV 轨迹 + 性能日志 + 可视化 + 位力平衡初值生成 + 参数扫描 + 回归断言框架 + 显存越界保护 + 质心/总动量自动归零**。

**实测**：在 RTX 5070 Ti Laptop（WSL2 + CUDA 12.2, CC=12.0, 46 SMs, 12GB）上：
- N=4096 × 1000 steps → ~400ms，Plummer 位力平衡能量漂移仅 **0.469%**（Leapfrog + dt=1e-3 + eps=0.02）
- **N=65536 × 1000 steps → 17.35s**，吞吐 **3.78 M particle-steps/s**，二进制 8.5MB 校验通过。

---

## 📦 环境要求
| 组件 | 最低要求 | 推荐 |
|------|---------|------|
| OS | Linux x86_64 / WSL2 | Ubuntu 20.04+ / WSL2 Ubuntu |
| NVIDIA Driver | ≥ 450.80 (支持 CUDA 11) | ≥ 550 支持 CUDA 12+ |
| CUDA Toolkit | 11.8+ | 12.2+ |
| CMake | 3.14+ | 最新版 |
| 编译器 | g++ ≥ 8 | g++ ≥ 10 |
| Python (脚本/可视化) | 3.8+ | 3.10+（numpy + matplotlib 必备） |

---

## 🚀 构建

本项目 `build.sh` 会**自动探测 CUDA 位置**（按优先级：`$CUDA_HOME` → `$(which nvcc)` 反推 → `/usr/local/cuda*` 常见目录），不需要手动改路径：

```bash
# 1) 首次 / 发布构建 (推荐, 全量构建)
bash build.sh --clean

# 2) 日常增量构建 (开发用)
bash build.sh

# 3) 打包发布 (可选, 默认 dist/ 目录)
cd build && cmake --install . --prefix ../dist
cd ..
# 产物:
#   dist/bin/nbody
#   dist/share/nbody/scripts/*.py  *.sh
#   dist/share/nbody/data/*.txt
#   dist/share/doc/nbody/{README,report,progress_report.csv}
```

构建成功后会看到：
```
---- build done ----
-rwxr-xr-x ... ./build/nbody
OK: ./build/nbody
```

### 构建失败诊断
`build.sh` 失败时会自动打印：
- `$CUDA_HOME` 是否设置
- `which nvcc` 的结果
- `/usr/local/cuda*` 目录列表
- `ldconfig -p | grep cuda` 前 8 行

常见解法：
1. 显式指定 `export CUDA_HOME=/usr/local/cuda-12.2 ; bash build.sh`
2. 安装缺少的依赖：`sudo apt-get install -y cmake build-essential`
3. 重装 CUDA Toolkit（确保 `$CUDA_HOME/bin/nvcc` 存在）。

---

## ⚡ 快速开始 (4 条命令跑通全流程)

```bash
# 1. 2 体稳定圆轨道 (正确性验证, 能量漂移 <0.1%)
./build/nbody data/particles_2body.txt data/params_default.txt \
    outputs/t2.bin outputs/perf.log \
    --check-energy --integrator leapfrog --dt 0.002 --G 1 --softening 0.01 \
    --steps 1000 --record 100

# 2. 3 体 + CSV 输出 (写-读可视化闭环)
./build/nbody data/particles_3body.txt data/params_default.txt \
    outputs/t3.bin outputs/perf.log \
    --csv --integrator leapfrog --steps 500 --record 50

# 3. 4096 粒子推荐档 (漂移最小 + 速度快)
./build/nbody data/particles_4096_plummer.txt data/params_default.txt \
    outputs/t4k.bin outputs/perf.log \
    --kernel tiling --softening 0.02 --dt 0.001 --steps 1000 --record 100 \
    --check-energy

# 4. 进阶版 65536 粒子 × 1000 步 (题面进阶要求, 约 17s)
./build/nbody data/particles_65536.txt data/params_default.txt \
    outputs/65k.bin outputs/perf.log \
    --kernel tiling --no-cpu --softening 0.02 --steps 1000 --record 100
```

---

## 🖥️ CLI 参数手册

### 位置参数 (顺序固定)
```
./build/nbody  <particles.txt>  <params.txt>  <trajectory.bin>  <perf.log>
```
| # | 作用 | 默认值 |
|---|------|--------|
| 1 | 粒子初值 7 列文件 (x y z vx vy vz mass) | `data/particles_default.txt` |
| 2 | 模拟参数键值对文件 | `data/params_default.txt` |
| 3 | 输出二进制轨迹 | `outputs/trajectory.bin` |
| 4 | 输出追加性能 JSON 行 | `outputs/perf.log` |

### 可选 flag (CLI 优先级永远高于 params.txt)
| 参数 | 说明 |
|------|------|
| `--info` | 仅打印 GPU 信息字符串后退出 (可测试 CUDA 环境) |
| `--kernel {simple,tiling}` | 加速度核函数 (tiling 大 N 更推荐) |
| `--integrator {euler,leapfrog}` | 积分器 (leapfrog 辛积分, 默认, 长期稳定) |
| `--dt FLOAT` | 每步时间步长 |
| `--steps INT` | 总步数 |
| `--record INT` | 每 INT 步记录一帧到轨迹 |
| `--G FLOAT` | 引力常数 (默认 1.0) |
| `--softening FLOAT` | 软化因子 ε (越大越"软"避免数值爆炸, 4096 推荐 0.02) |
| `--block-size INT` | CUDA block 大小 (256 默认) |
| `--no-cpu` | 跳过 CPU 基准 (节省时间, 无 speedup 输出) |
| `--check-energy` | 每 record_interval 步同步回主机算总能量/动量并打印漂移 |
| `--csv` | 额外写出 trajectory.bin.csv 格式轨迹 |
| `--analyze N` | 只分析前 N 帧 (截断 steps, 用于快速 preview) |

参数覆盖优先级：**CLI flag > params.txt 内同名键 > 代码默认值**。

---

## 🧪 生成新的粒子初值 (3 种模型)
`scripts/generate_particles.py` 支持 **Plummer 球 / 均匀位力球 / 开普勒盘**，并自动做质心 + 总动量归零：

```bash
# Plummer 65536 粒子 (推荐用于进阶版)
python3 scripts/generate_particles.py plummer 65536 data/p_65k.txt --seed 2026 --Rscale 2.0 --Mtot 10.0

# 均匀位力平衡球 4096 (代替原 v=0 冷球, 显著降低能量漂移)
python3 scripts/generate_particles.py sphere 4096 data/p_4k_sphere.txt --Rscale 3.0 --eps 0.02

# 薄星系盘 8192 (可视化看漩涡结构)
python3 scripts/generate_particles.py disk 8192 data/disk_8k.txt --Rscale 1.0
```

> 💡 所有生成文件在加载到 `main.cpp` 后还会**再做一次质心+动量归零**（见下方"数值稳健性"），跨脚本/跨平台始终一致。

---

## 🔎 参数扫描找最佳能量漂移
对 4096 Plummer 做 25 组 `dt × eps` 全组合扫描，自动找漂移最小档：
```bash
python3 scripts/scan_drift.py
```
典型输出（示例）：
```
dt=1e-4 eps=0.1   dE/E0% = 0.0000% (最佳)
dt=1e-3 eps=0.02  dE/E0% = 0.469%  (平衡档, 推荐)
dt=3e-3 eps=0.1   dE/E0% = 4.965%
```
结果直接拿来贴到 `--dt` / `--softening`。

---

## 📊 性能日志转 CSV / 画图
`outputs/perf.log` 每行是一个 JSON（多次运行自动追加，便于历史对比）。转成带列名的 CSV：
```bash
python3 scripts/perf_to_csv.py outputs/perf.log -o outputs/perf_summary.csv
# 打印列定义
python3 scripts/perf_to_csv.py --schema
# 按吞吐倒序取前 5
python3 scripts/perf_to_csv.py outputs/perf.log --sort throughput_particles_per_s --descend -n 5
```
字段包括：`mode, integrator, kernel, N, steps, dt, eps, total_ms, avg_step_ms, throughput_particles_per_s, gpu_mem_MB, cpu_time_ms, speedup, traj_file, particles_file, comment`…

---

## 🎨 可视化 (2D/3D, GIF/MP4, bin/csv/demo)
`scripts/visualize.py` 是独立脚本，**无需 CUDA 就能跑**（只要 NumPy + Matplotlib）：
```bash
# 二进制轨迹 2D GIF (默认)
python3 scripts/visualize.py outputs/t2.bin --out outputs/t2.gif

# 65k 粒子 3D 预览: 下采样到 8000 点，避免 matplotlib 卡顿 (--max-points)
python3 scripts/visualize.py outputs/65k.bin --view 3d --frames 11 \
        --max-points 8000 --out outputs/65k.gif --fmt gif

# CSV 轨迹 (visualize 自动根据后缀识别)
python3 scripts/visualize.py outputs/t3.bin.csv --out outputs/t3_fromcsv.gif

# 纯内置 Demo 模式 (不需要仿真产物, 教学用)
python3 scripts/visualize.py demo 2body
python3 scripts/visualize.py demo 3body
python3 scripts/visualize.py demo cluster --out outputs/cluster_demo.gif

# MP4 (需要 ffmpeg, 比 GIF 小 80%)
python3 scripts/visualize.py outputs/65k.bin --fmt mp4 --out outputs/65k.mp4
```

---

## ✅ 回归测试 (两套)

开发阶段快速验证（100 步冒烟，秒级）：
```bash
bash run_test.sh
# 输出 4 节: build -> 2body -> 3body -> 4096 smoke
# 最后 "all smoke runs ok"
```

发布 / 提交前严格断言（1000 步，百分比/尺寸/动量**有明确阈值**，FAIL 会 exit 1）：
```bash
bash regression_test.sh
```
覆盖断言项：
| # | 测试 | 阈值 |
|---|------|------|
| 0 | build 成功 | 必过 |
| 1.1 | 2body 二进制头+大小匹配 | N=2, R≥11 |
| 1.2 | 2body step-0 动量 | \|P\| < 1e-10 |
| 1.3 | 2body 能量漂移 (1000 步) | < 0.1% |
| 2.1 | 3body 二进制匹配 | N=3, R≥10 |
| 2.2 | 3body CSV 输出 | 文件存在 >1KB |
| 2.3 | 3body 能量漂移 (500 步) | < 2% |
| 3.1 | 4096_plummer 二进制匹配 | N=4096, R≥11 |
| 3.2 | 4096_plummer 能量漂移 (1000 步 tiling+dt=1e-3 eps=0.02) | < 1% |
| 3.3 | 4096_plummer 吞吐 | > 8 M particle-steps/s |

断言实现：[`scripts/regression_assert.py`](scripts/regression_assert.py)（支持 `bin_ok / momentum_step0 / rel_number / line_present` 4 种模式）。

---

## 🛡️ 数值稳健性 & 越界保护
### 1) 质心 (CM) + 总动量自动归零
`main.cpp` 在 `parse_particles_file` 成功后用 **double 累加** 重新计算质心位置和质心系速度，对 float 粒子数组做平移，写入日志：
```
[Sim] CM+P zeroed: Mtot=10.000  |P|_before=0.000  CM_before=(0.000,-0.000,0.000)
```
避免因 `particles.txt` 生成脚本版本差异、10 位小数截断、或人为手动编辑导致整体漂移（这对长时程仿真非常关键）。

### 2) GPU 显存预估 + 友好错误
在 `cudaMalloc` 之前调用 `nbody_gpu_memory_ok_c`（基于 `cudaMemGetInfo`），打印估算值：
```
[VRAM] OK: N=65536 need~4MB, GPU free=11026MB / total=12226MB (headroom=11022MB)
```
不足时直接 `exit 11` + 中文错误（提示关闭 Chrome/IDE/WebGL/WebGPU 等占用 GPU 的程序）。

### 3) 二进制轨迹严格格式
```
Header: int32 N, int32 R              (8 字节)
Data  : R 帧 × { N × (float32 x, float32 y, float32 z) }
总字节: 8 + 12*N*R
```
用 `scripts/check_bin.py` 随时校验：
```bash
python3 scripts/check_bin.py outputs/65k.bin
# outputs/65k.bin: N=65536 R=11 bytes=8650760 expected=8650760 OK=True
```

---

## 📂 目录结构
```
CudaNBodyGravitySim_2026summer/
├── src/                         实现
│   ├── main.cpp                 CLI + 主循环 + 计时 + CM/P归零 + 显存保护
│   ├── nbody_kernels.cu         CUDA kernel (simple / shared_tiling) + integrator + extern-C host wrapper
│   ├── file_io.cpp              particles/params 解析 + 二进制 / CSV / perf.log 写出
│   └── analysis.cpp             CPU O(N²) 参考积分 + 动能/势能/动量/能量计算
├── include/                     头文件 (无 CUDA 依赖, C++ TU 可直接 include)
│   ├── nbody_types.h            ParticleSet / SimParams / Integrator / KernelMode
│   ├── nbody_kernels.h          GPU wrapper 声明 (extern "C")
│   ├── file_io.h
│   └── analysis.h
├── scripts/
│   ├── detect_sm.sh             CUDA SM 3 策略探测 (compute_cap 查表→GPU名→默认列表)
│   ├── generate_particles.py    3 种模型粒子生成 (Plummer/sphere/disk)
│   ├── scan_drift.py            dt×eps 参数扫描 (寻找能量漂移最优档)
│   ├── visualize.py             轨迹可视化: bin/csv/demo, 2d/3d, gif/mp4, 下采样
│   ├── check_bin.py             二进制轨迹 header+frame 字节对齐校验
│   ├── perf_to_csv.py           perf.log JSON → CSV (Pandas/Excel 友好)
│   └── regression_assert.py     4 种断言模式供 regression_test.sh 调用
├── data/                        particles / params 样例
│   ├── particles_{2body,3body,4096,4096_plummer,65536}.txt
│   └── params_{default,fast}.txt
├── outputs/                     仿真产物 + regression 日志
│   ├── regression/              regression_test.sh 独立输出目录 (每个测试 1 bin + 1 log + summary)
│   └── *.bin / *.csv / *.log / *.gif
├── CMakeLists.txt               (含 install 打包规则: bin + scripts + docs + data)
├── build.sh
├── run_test.sh                  (smoke 100 步开发版)
├── regression_test.sh           (1000 步严格断言发布版)
├── README.md                    (你正在读的)
├── report.md                    详细总结报告 (算法/性能/参数扫描/公式)
└── progress_report.csv          项目级交付状态 CSV (每阶段实际数据回填)
```

---

## 📖 延伸阅读
- 详细实现报告 / 参数扫描表 / 65k 实测细节 → [`report.md`](report.md)
- 交付进度跟踪（完成度+实测值）→ [`progress_report.csv`](progress_report.csv)
- `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` → 决定你 GPU 的 SM 代号。

---

## ❓ FAQ
**Q1: 2 体能量漂移 > 0.1%？** → 99% 是 `dt` 太大或 `eps` 太小。改：`--dt 0.001 --softening 0.01 --integrator leapfrog --steps 1000`。
**Q2: 4096 粒子能量漂移 > 10%？** → 99% 你在用 v=0 冷球（自由下落坍缩天然能量剧烈变动）。改用 `particles_4096_plummer.txt`（位力平衡），再配 `--dt 1e-3 --softening 0.02` 即可降到 <1%。
**Q3: 65k 跑"GPU VRAM 不足"？** → 你可能运行了其它吃 GPU 的程序（Chrome 60+ 标签 + IDE + 视频会议通常占 2~4GB）。最简单重启 / `nvidia-smi` 看谁占了，关了再跑。
**Q4: visualize.py 巨慢（65k 几分钟还不出 GIF）？** → 一定加 **`--max-points 8000`**（matplotlib scatter 65k 每帧巨慢，下采样到 8k 观感上完全一样）。
**Q5: AutoDL GPU 上跑需要改什么？** → 不需要。`bash build.sh` 会自动 `which nvcc` 探测；`detect_sm.sh` 直接读 `nvidia-smi compute_cap`（A100 → sm_80、A10 → sm_86、L40 → sm_89、H100/B100 → sm_90 PTX JIT）。
**Q6: 如何在别的机器上复现？** → `bash build.sh` 会自动 `which nvcc` 探测 CUDA 并调用 `scripts/detect_sm.sh` 决定 `-gencode`；`run_test.sh` 跑 100 步 smoke，`regression_test.sh` 跑 1000 步严格断言并输出到 `outputs/regression/`。
