# README And CMake Presets Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为项目补充面向 WSL + Conda 的 `CMakePresets.json` 和完整 `README.md`，并附带云服务器兼容说明。

**Architecture:** 通过 `CMakePresets.json` 统一 Conda 工具链入口，使用 `README.md` 作为单一用户指南，覆盖环境、构建、运行、测试、数据格式与兼容说明。方案保持源码行为不变，只增强配置与文档。

**Tech Stack:** CMake Presets、Ninja、CUDA Toolkit、OpenCV、Markdown

---

### Task 1: 新增 CMake 预设

**Files:**
- Create: `CMakePresets.json`
- Modify: `CMakeLists.txt`

**Step 1: 确认 `CMakeLists.txt` 导出 `compile_commands.json`**

检查 `CMakeLists.txt` 中存在：

```cmake
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
```

**Step 2: 编写 `CMakePresets.json`**

添加：
- `conda-release`
- `conda-debug`
- `build-conda-release`
- `build-conda-debug`
- `build-test-correctness`
- `build-test-performance`

要求：
- 默认依赖 `$env{CONDA_PREFIX}`
- 配置 `CUDAToolkit_ROOT`、`OpenCV_DIR`、`CMAKE_PREFIX_PATH`
- 输出到 `build/conda-release` 和 `build/conda-debug`

**Step 3: 验证 release preset 可配置**

Run: `cmake --preset conda-release`

Expected: 成功生成 `build/conda-release`

---

### Task 2: 编写 README

**Files:**
- Create: `README.md`

**Step 1: 编写项目简介与目录说明**

要求：
- 解释项目目标
- 概述三个 kernel 阶段
- 给出目录结构

**Step 2: 编写环境配置说明**

要求：
- 明确主环境为 WSL2 + Ubuntu 22.04 + Conda
- 给出 Conda 环境创建命令
- 给出激活命令与基础验证命令

**Step 3: 编写构建与运行说明**

要求：
- 优先使用 `cmake --preset conda-release`
- 给出 `cmake --build --preset build-conda-release`
- 给出主程序、正确性测试、性能测试运行命令

**Step 4: 编写数据格式与参数说明**

要求：
- raw 文件格式
- `params.txt` 格式
- kernel 选择参数说明

**Step 5: 编写云服务器兼容说明与 FAQ**

要求：
- 说明如何复用同一套 Conda 依赖
- 说明 WSL 与普通 Linux 的差异点
- 包含 `nvcc not found`、includePath、GPU 运行限制等常见问题

---

### Task 3: 验证文档与预设一致

**Files:**
- Verify: `CMakePresets.json`
- Verify: `README.md`

**Step 1: 用 preset 重新配置**

Run: `cmake --preset conda-release`

Expected: 成功

**Step 2: 用 preset 构建主目标**

Run: `cmake --build --preset build-conda-release`

Expected: 成功生成 `bilateral_filter`、`test_correctness`、`test_performance`

**Step 3: 人工检查 README 命令与 preset 名称一致**

核对：
- 预设名称
- 构建目录
- 环境变量说明
- 运行命令

**Step 4: 总结并汇报**

记录：
- 新增文件
- 验证命令
- 任何仍需用户手动执行的步骤
