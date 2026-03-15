# 09_particle_sim — nagadoyuji

带电粒子在磁场中的运动轨迹 CUDA 模拟，输出轨迹供 Python 3D 可视化。

- **作者 / 提交 ID**：nagadoyuji
- **总结报告**：[总结报告.md](总结报告.md)

## 快速开始

```bash
make
./particle_sim data/particles.txt data/field.txt data/params.txt out.bin
python visualize.py out.bin
```

## 目录说明

| 文件 | 说明 |
|------|------|
| main.cu | CUDA 主程序（解析、Boris 积分、记录、二进制输出） |
| Makefile | 构建（`make` / `make format` 格式化） |
| .clang-format | 代码格式配置 |
| data/ | 粒子、磁场、参数示例 |
| visualize.py | 3D 轨迹动画（Matplotlib FuncAnimation） |
| verify_single_particle.py | 单粒子回旋半径正确性验证 |
| generate_particles.py | 生成 1000 粒子等规模测试数据 |
| generate_B_grid.py | 生成 3D 磁场网格二进制文件 |

## 正确性验证

```bash
./particle_sim data/particles_1.txt data/field.txt data/params_verify.txt out.bin
python verify_single_particle.py out.bin data/particles_1.txt data/field.txt
```

## 规模测试（≥1000 粒子）

```bash
python generate_particles.py -n 1000 -o data/particles_1k.txt
./particle_sim data/particles_1k.txt data/field.txt data/params.txt out.bin
```

## 环境

- CUDA Toolkit 11+，C++17
- Python 3：numpy, matplotlib
