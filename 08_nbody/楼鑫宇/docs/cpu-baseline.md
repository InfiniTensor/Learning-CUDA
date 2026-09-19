# CPU FP64 基线实验

## 实验目的

这组实验确认项目骨架可以完成“读取初态—数值积分—记录轨迹—输出诊断—Python 读回”的完整流程，并为后续 CUDA 实现提供正确性参考和速度基线。这里的数据是本项目服务器上的真实首轮结果，不是题目规定的性能阈值。

## 实验环境

- 运行日期：2026-09-13
- CPU：Intel Xeon Platinum 8358P @ 2.60 GHz
- 编译器：GCC 13.3.0
- CMake：3.31.4
- 构建类型：Release
- CPU 实现：单线程、FP64、全粒子直接求和
- 积分器：Leapfrog KDK

CPU 型号来自宿主机信息。当前程序是单线程参考版本，因此宿主机显示的总逻辑 CPU 数量不会直接提高这次运行速度。

## 二体轨道验证

输入为两个等质量粒子，初始间距为 1，使用 `configs/two_body.cfg` 运行 10,000 步。

| 指标 | 实测值 |
|---|---:|
| 力计算耗时 | 0.489 ms |
| 模拟耗时 | 0.842 ms |
| 最大相对间距误差 | 9.537e-7 |
| 最大质心距离 | 0 |
| 相对能量误差 | 1.693e-13 |
| 坐标是否全部有限 | 是 |

轨迹以 float32 保存，所以从轨迹文件计算的间距误差会包含输出量化误差；能量误差由 CPU FP64 最终状态计算。

## 4096 粒子、1000 步基线

初态由 `scripts/generate_cluster.py` 使用固定随机种子 42 生成，配置为 `configs/cpu_4096.cfg`。输出包含初始状态、每 10 步一次的状态和末状态，共 101 帧。

| 指标 | 实测值 |
|---|---:|
| 粒子数 | 4096 |
| 模拟步数 | 1000 |
| 力计算次数 | 1001 |
| 力计算总耗时 | 67,586.864 ms |
| 模拟总耗时 | 67,608.592 ms |
| particle-steps/s | 60,584.016 |
| 相对能量误差 | 2.307e-6 |
| 绝对动量误差 | 5.153e-17 |
| 轨迹记录数 | 101 |
| 轨迹文件大小 | 4,964,360 字节 |

1001 次力计算来自 Leapfrog 的初始加速度计算和每一步末尾的新加速度计算。这里的 `particle-steps/s` 以 1000 个积分步计数，不把额外一次初始力计算算作积分步。

## 复现命令

```bash
source /data/nbody-dev/activate-nbody.sh
cd /data/nbody-project/08_nbody/楼鑫宇

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure

./build/nbody --backend cpu \
  --input data/two_body.txt \
  --config configs/two_body.cfg \
  --output results/two_body_cpu
python scripts/validate_two_body.py results/two_body_cpu

python scripts/generate_cluster.py \
  --n 4096 --seed 42 --output data/cluster_4096.txt
./build/nbody --backend cpu \
  --input data/cluster_4096.txt \
  --config configs/cpu_4096.cfg \
  --output results/cluster_4096_cpu_1000
```

每次运行应使用新的输出目录名。`results/` 和生成的 `data/cluster_*.txt` 已加入 `.gitignore`，不会把大型实验产物误提交到仓库。

## 后续比较规则

CUDA 版本先用少量粒子和少量步数逐粒子对照 CPU FP64 结果，再运行 4096 粒子、1000 步。速度比较需要保持初态、时间步、软化长度、积分器和轨迹记录策略一致；GPU 需预热后重复运行并报告中位数。GPU FP32 与 CPU FP64 的对比属于实现速度比较，报告中还需单独给出误差，不能把精度差异隐藏在加速比中。
