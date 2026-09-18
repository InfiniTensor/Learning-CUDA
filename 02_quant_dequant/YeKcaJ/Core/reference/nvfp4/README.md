# CPU NVFP4 参考实现

这是选题 2 的 CPU reference，用来定义 NVFP4 的确定性量化结果，后续
CUDA packed kernel 必须以它作为正确性对照。

## 当前格式

- 每个 block 包含 16 个元素。
- 每个元素使用 E2M1 4 bit，幅值为 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`，最高 bit 保存符号。
- 一个字节保存两个元素：偶数元素放低 4 bit，奇数元素放高 4 bit。
- 每个 block 保存一个 E4M3 block scale，整个张量保存一个 FP32 global scale。
- 有效缩放因子：`effective_scale = global_scale * block_scale`。
- 设整个张量最大绝对值为 `M`，则
  `global_scale = M / (6 * 448)`；全零张量使用 1。
- 设 block 最大绝对值为 `m`，则 block scale 量化为
  `E4M3(m / (6 * global_scale))`。
- 量化采用最近值舍入，反量化公式为
  `decode_e2m1(nibble) * effective_scale`。

## 构建

以下 CPU 专用命令在 `Core/reference/nvfp4` 内执行。正常使用 CUDA 程序只需构建 Core，无需单独构建本目录。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## 运行模式

运行基本自测：

```bash
./build/cpu_nvfp4 --self-test
```

普通演示会生成随机 FP32 矩阵并写出 packed 量化文件：

```bash
./build/cpu_nvfp4 1024 1024 nvfp4.bin
```

## 冻结 CPU reference

冻结的作用是建立一套固定的“标准答案”，让后续 CUDA 版本使用完全相同
的输入，并检查 packed 字节、scale 和反量化结果。它不是性能测试，也不是
要求量化误差为零。

现有标准文件已经冻结，日常只运行下方 `--verify`。首次建库时使用的命令为：

```bash
./build/cpu_nvfp4 --freeze
```

该命令生成：

- `tests/data/`：固定 FP32 输入；
- `tests/golden/*.nvfp4`：固定 packed data、block scales 和 global scale；
- `tests/golden/*.dequant.fp32`：固定反量化 FP32 结果；
- `tests/FREEZE_TEST_RESULTS.txt`：测试结果和误差；
- `tests/golden/SHA256SUMS.txt`：输入和 golden 文件哈希。

冻结完成后，日常验证使用：

```bash
./build/cpu_nvfp4 --verify
sha256sum -c tests/golden/SHA256SUMS.txt
```

`--verify` 不会覆盖已有 golden。如果 E2M1 编码、scale 公式或 packed 布局
被修改，验证应失败；只有明确升级格式时才重新运行 `--freeze`，并提升版本号。
