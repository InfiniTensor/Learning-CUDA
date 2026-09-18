# CPU MXFP8 参考实现

这是选题 2 的 CPU reference，用来定义 MXFP8 的确定性量化结果，后续
CUDA kernel 必须以它作为正确性对照。

## 当前格式

- 数据格式：E4M3FN，每个元素 1 字节。
- block 大小：32 个元素。
- 每个 block 保存一个 E8M0 缩放指数，偏置为 127。
- 设 block 最大绝对值为 `m`，指数为
  `clamp(ceil(log2(m / 448)) + 127, 0, 254)`。
- 实际缩放因子为 `2^(指数 - 127)`；全零 block 使用 scale 1。
- 量化采用最近值舍入，范围外数值饱和到 `[-448, 448]`。
- 反量化公式：`decode_e4m3(code) * scale`。

## 构建

以下 CPU 专用命令在 `Core/reference/mxfp8` 内执行。正常使用 CUDA 程序只需构建 Core，无需单独构建本目录。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## 运行模式

运行基本自测：

```bash
./build/cpu_mxfp8 --self-test
```

普通演示会生成随机 FP32 矩阵并写出量化文件：

```bash
./build/cpu_mxfp8 1024 1024 mxfp8.bin
```

## 冻结 CPU reference

冻结的作用是建立一套固定的“标准答案”，让后续 CUDA 版本使用完全相同
的输入，并逐字节、逐元素对照。它不是性能测试，也不是把 FP32 误差变成零。

现有标准文件已经冻结，日常只运行下方 `--verify`。首次建库时使用的命令为：

```bash
./build/cpu_mxfp8 --freeze
```

该命令生成：

- `tests/data/`：固定 FP32 输入；
- `tests/golden/*.mxfp8`：固定量化 data 和 block scales；
- `tests/golden/*.dequant.fp32`：固定反量化 FP32 结果；
- `tests/FREEZE_TEST_RESULTS.txt`：测试结果和误差；
- `tests/golden/SHA256SUMS.txt`：输入和 golden 文件哈希。

冻结完成后，日常验证使用：

```bash
./build/cpu_mxfp8 --verify
sha256sum -c tests/golden/SHA256SUMS.txt
```

`--verify` 不会覆盖已有 golden。如果量化公式、scale 或文件布局被修改，
验证应失败；只有明确升级格式时才重新运行 `--freeze`，并提升版本号。
