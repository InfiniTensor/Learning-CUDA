# Luxorion 沐曦 C500 适配工作区

本目录是 `Luxorion` 的沐曦曦云 C500（MXMACA）适配副本，并在同一块 C500、同一组输入和同一计时协议下重新执行八个优化方向。NVIDIA 上的历史结论只作为参考，不直接继承。

## 实施顺序

1. 冻结 CPU reference、输入文件、量化格式、误差指标和输出哈希，作为跨平台 oracle。
2. 建立 `v0_metax_baseline`：使用 MXMACA 工具链编译现有 CUDA 风格源码，替换不兼容的编译选项、运行时 API、设备查询和计时/构建逻辑。
3. 在 C500 上逐项验证 MXFP8 / NVFP4 的量化、反量化、尾块、FP16/FP32 输入及 CLI 全流程；先保证结果契约一致，再记录 baseline 性能。
4. 依次实验八个方向，每个方向保留独立源码、命令、环境快照、正确性结果和性能结果。只有优于上一最佳版本且核心用例不回退的候选才晋级。

## 当前构建入口

在 C500 镜像内执行 `bash scripts/build_metax.sh`。脚本先用 CMake 构建独立 CPU reference，再用 MXMACA `mxcc` 编译 GPU baseline；不调用 CMake 的 CUDA 模块，因此不要求 `nvcc`。构建生成 `build/metax/luxorion_metax_cli`（冻结 v0 baseline）及 `build/metax/luxorion_metax_v7`（当前 C500 最佳版本）。

## 八个优化方向

1. Thread / Block 配置与 occupancy。
2. 全局内存合并访问。
3. 向量化加载与存储。
4. Warp shuffle / 子组归约。
5. Shared memory 与 scale 数据复用。
6. Shared-memory bank conflict。
7. 分支、查表与编码搜索路径优化。
8. ILP、循环展开与指令调度。

## 服务器首次连接需要采集

- `uname -a`、`cat /etc/os-release`
- `mx-smi` 完整输出（驱动、MACA、显存和 sGPU 配额）
- `mxcc --version` 或镜像中实际编译器路径与版本
- `cmake --version`、`ninja --version`、`gcc --version`
- `mxcc` 是否已加入 `PATH`，以及 `MXCC_BIN`、`MACA_PATH`、`MACA_INCLUDE` 等环境变量
- C500 是否为整卡或 sGPU；若为 sGPU，计算比例与显存配额

## 镜像原则

优先使用模力方舟平台提供、明确支持 C500 且内置 MXMACA C/C++ 编译工具链的 Ubuntu 22.04 x86_64 镜像。镜像内 MACA 用户态版本必须与宿主机驱动兼容；不要先选纯 Ubuntu、纯 PyTorch 推理或仅有 Python 框架而缺少 `mxcc`/开发头文件的镜像。

当前目录仅建立迁移计划。收到服务器地址和密钥并确认工具链后，再把 `Luxorion` 源码复制/改造为可编译的 v0 baseline，避免在本地假设沐曦编译器行为或写入未经 C500 验证的性能结论。

## 最终交付目录

沐曦 C500 的最终文件统一整理在 `gpu_final/metax_c500/`。其中 `source/` 保存当前最佳 v7 源码，`compat/` 保存 MXMACA 兼容层，`scripts/` 保存相对路径构建与复测入口，`results/` 保存 C500 实测数据。
