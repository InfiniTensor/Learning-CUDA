# Hadamard 变换加速（Tensor Core）

作业 03 的个人实现工程。目标：在 RTX 4090（sm_89）上实现沿最后一维的 Hadamard 变换，
优先使用 Tensor Core，并把量化 epilogue 融合进同一 kernel。

- 输入：4D 张量 `[batch_size, seq_len, num_heads, head_dim]`，FP16 或 BF16
- `head_dim` 为 2 的幂（64 / 128 / 256 等），非 2 幂尺寸不在范围内
- 输出：变换结果 + kernel 执行时间（ms）
- 正确性门槛：FP16 绝对误差 < 1e-2，BF16 < 5e-2

## 变换约定

统一采用 `y = H_d * x / sqrt(d)`，其中 `H_d` 为 Sylvester 矩阵
（`H_1 = [1]`，`H_2n = [[H_n, H_n], [H_n, -H_n]]`）。`1/sqrt(d)` 使变换正交归一，
因此连续施加两次等于恒等变换——对拍框架直接利用了这条性质。

该约定已与参考实现 `fast_hadamard_transform` 对齐并做了数值验证（d = 2…512、fp16/bf16
逐元素 bit-exact），详见 `docs/learning_notes.md` 的「已确认的约定」一节。

## 环境

| 项 | 值 |
|---|---|
| GPU | RTX 4090 D（Ada, sm_89, 24 GiB） |
| 驱动 / CUDA | 570.124.06 / CUDA 12.8 |
| nvcc | `/usr/local/cuda/bin/nvcc`（不在默认 PATH） |
| 构建 | cmake >= 3.24、make、g++ |

`nvcc` 不在 PATH 时先导出：

```
export PATH=/usr/local/cuda/bin:$PATH
```

## 构建与运行

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/hw_probe     # M0 探针：设备信息 + hello kernel + WMMA identity 探针
./build/hw_tests     # 对拍：参考实现互校、正交性、精度往返、GPU 实现对比
./build/hw_bench     # benchmark：kernel ms 与有效带宽 GB/s
```

也可以走 ctest：

```
cd build && ctest --output-on-failure
```

覆盖目标架构用 `-DCMAKE_CUDA_ARCHITECTURES=89`（默认已是 89）。

不需要自己拼命令：报告 `docs/report.md` 的 §8 列出了上面每条命令与报告里每张表的对应关系，以及
各份原始日志（`docs/logs/`）的生成方式。

## 目录结构

```
include/hadamard/   common.h（shape/dtype/误差门槛）、cuda_utils.h（错误检查/事件计时）、hadamard.h（公开 API）
src/hadamard_ref.cu CPU FP32 参考：Sylvester 显式 matmul + 蝶形 FWHT + FP16/BF16 打包解包
src/hadamard_gpu.cu GPU 入口：hello kernel、WMMA identity 探针、M2 寄存器蝶形 FWHT kernel、M3 Tensor Core（WMMA）kernel、M4 量化 kernel（独立量化 + 蝶形融合 + TC 融合）
src/probe_main.cu   M0 探针可执行文件
tests/              对拍框架（M2/M3 两条 GPU 路径各自与参考实现对拍）
bench/              benchmark 驱动：kernel ms + 有效带宽
docs/               学习笔记、总结报告与原始日志
  docs/report.md      交付报告（摘要 / 实现 / 优化历程 / 性能 / 验证 / 结论 / 复现方式 / 附录）
  docs/logs/          报告引用的原始日志：hw_bench stdout、-Xptxas -v、nsys 统计摘录、ctest 记录
```

## 进度

- [x] M0 工程骨架：CMake（sm_89）、clang-format、探针跑通
- [x] M1 CPU FP32 参考实现 + 多 shape 对拍框架（FP16/BF16）
- [x] M2 蝶形 FWHT baseline（`launch_fwht_baseline`）：寄存器内蝶形 + warp shuffle，
      不用 shared memory；10 个 GPU 用例全 PASS，DRAM 受限带宽 943 GB/s（峰值 93%）
- [x] M3 WMMA Tensor Core 主实现（`launch_hadamard_tc`）：把 `X · H_D` 写成 GEMM，K = D 按 16
      切分；靠 Kronecker 分块只常驻 ±H_16（不存 d×d 矩阵），累加器 FP32、epilogue 里缩放并写回。
      20 个 GPU 用例全 PASS。支持范围：`d ≥ 16`；`d ∈ {2,4,8}` 无法满足 `m16n16k16` 的 K ≥ 16，
      自动回落 M2。DRAM 受限的大 shape 与 M2 打平（935 vs 943 GB/s）；L2 常驻的 shape 因为算术
      强度升到 `D/2` 而变成计算受限，慢 1.5~2.5×（见 `docs/report.md` 第 4 节）。
- [x] M4 FP8 E4M3 per-token 量化融合（`launch_quantize_fp8` / `launch_fwht_baseline_fp8` /
      `launch_hadamard_tc_fp8`）：融合结果与「先变换后量化」两段式**逐字节一致**（20 个用例码字
      0 差异、scale 逐位相等）；DRAM 受限行 0.2138 ms vs 两段式 0.4996 ms（2.34×，与 7/3 的流量账
      吻合），蝶形融合直接跑在纯量化的带宽上（951 vs 951 GB/s）。支持范围：融合 TC 路径 `d ∈ [16,512]`，
      蝶形融合覆盖 `d = 2…1024`。
- [x] M5 全 shape benchmark + profiler 分析：benchmark 拆成两张表（6 个访存档位 + `d = 8…1024`
      扫描），每个 shape 都带一行同 footprint 的 `copy_kernel` 作为搬运上限；本容器的 `ncu` 不可用
      （`ERR_NVGPUCTRPERM`、`RmProfilingAdminOnly: 1`、容器内无 `CAP_SYS_ADMIN`），改用 `-Xptxas -v`
      的占用率表与 `nsys` 时间线作侧证。结论：M2 始终贴住搬运上限；M3 在 `d ≥ 256` 已达裸 mma 峰值
      的 84~90%，但仍比 M2 慢（`D²` 对 `D·log2 D` 的 MAC 差，交叉点约 `d = 110`）；融合量化上蝶形版
      全面优于 TC 版（`d ≥ 16` 快 2.0~6.8×）。见 `docs/report.md` §4。
- [x] M6 报告：`docs/report.md` 补完为终稿（摘要、§1 验收口径与硬件表、§7 结论、§8 复现方式、
      附录 A 原始日志 / 附录 B 提交记录）；报告里每个数字都能在 `docs/logs/` 里核到原文
- [x] M7 整理与 PR：全部源码 `clang-format --dry-run` 0 处违规、全新 Release 构建 0 warning；补充归档
      `ctest` 原始记录（`docs/logs/ctest_rtx4090d.txt`）；以 PR 提交至 `InfiniTensor/Learning-CUDA` 的
      `2026-summer-project` 分支，路径 `03_hadamard_tc/MarinusZhang/`

GPU 路径与 M4 融合量化均已落地：`hw_tests` 共 57 个检查（其中 45 个涉及 GPU），0 failed / 0 skipped；
`ctest` 2/2（0.57 s）。M5 改完 benchmark 后重跑无回归。

## 对拍与 benchmark 约定

- 输入先按目标精度取整，参考实现作用在取整后的输入上，因此误差只反映 kernel 误差，
  不混入输入量化误差。
- 有效带宽按一次读 + 一次写统计（FP16/BF16 为 4 字节/元素）。量化路径按真实流量计：融合版为
  「读一次激活 + 写一次 FP8 码字」= 3 B/元素（另加每行 4 B 的 scale），两段式为 7 B/元素，
  因此两边都贴住带宽上限时融合应得 7/3 ≈ 2.33×。
- benchmark 使用 CUDA event 计时，GPU 侧 50 次迭代求均值（含 1 次预热）。
- 小 shape 的工作集在重复迭代中驻留 L2，其 GB/s 反映 L2 带宽；最后一个形状（256 MiB 流量）
  超出 72 MiB L2，用来衡量 HBM 受限下的表现。
- 每个 shape 先跑一行同 footprint 的 `copy_kernel`（uint4 grid-stride 的一次读 + 一次写）作为该
  尺寸下的搬运上限；某一行报出的带宽与它相同时，说明这行已经贴在访存上限上而不是「慢」。融合行
  只搬 3 B/元素，因此它的**时间**下界是 copy 的 3/4，而**带宽**上限与 4 B/元素的行相同。
- CPU 参考行只在元素数不超过 `64 · 1024 · 1024` 时测量：`ref_fwht_fp32` 是 `O(d·log2 d)` 的蝶形
  实现，代价随元素数线性增长，与 `d²` 无关。
- 表 2（head_dim 扫描）把元素数固定为 4 Mi（16 MiB 流量，留在 L2 内）以隔离 `d` 的影响；表中
  `rows = 1` 的行是延迟探针（GB/s 列没有意义），`d = 8` 的 TC 行是蝶形回落、`d = 1024` 的融合 TC
  行不支持，均按实际行为标注。

## 参考资料

- Tri Dao, *fast-hadamard-transform*（QuaRot 同款 kernel）
- QuaRot / SpinQuant：量化前旋转抑制异常值
- CUDA Samples `simpleTensorCoreGEMM`（WMMA API 用法）
