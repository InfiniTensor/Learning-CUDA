# README 与 CMake Presets 设计文档

## 文档信息

- 日期：2026-03-07
- 范围：为 `bilateral_filter` 项目补充 `CMakePresets.json` 与 `README.md`
- 主环境：WSL2 + Ubuntu 22.04 LTS + Conda

---

## 一、目标

- 为项目提供可复用的 `CMakePresets.json`，简化 WSL + Conda 环境下的配置、构建与调试。
- 提供完整的 `README.md`，覆盖环境配置、编译运行、测试验证、输入输出格式与云服务器兼容说明。
- 保持项目源码行为不变，仅增强构建入口与使用文档。

## 二、设计原则

- **WSL + Conda 优先**：文档和预设默认服务当前主开发环境。
- **云服务器兼容**：使用同一套 Conda 依赖时，无需修改项目代码。
- **低耦合**：预设使用 `$env{CONDA_PREFIX}`，避免硬编码单机绝对路径。
- **编辑器友好**：保留 `compile_commands.json` 导出，方便 VS Code / clangd 补全。
- **最少惊讶**：默认输出到独立构建目录，避免与已有临时构建目录混淆。

## 三、CMake Presets 设计

### 配置预设

- `conda-release`
  - 生成器：`Ninja`
  - 构建目录：`build/conda-release`
  - 类型：`Release`
  - 依赖当前已激活的 Conda 环境，通过 `$env{CONDA_PREFIX}` 注入：
    - `CUDAToolkit_ROOT`
    - `OpenCV_DIR`
    - `CMAKE_PREFIX_PATH`
    - `CPATH`
    - `LIBRARY_PATH`
    - `LD_LIBRARY_PATH`
- `conda-debug`
  - 继承 `conda-release`
  - 仅切换到 `Debug`
  - 构建目录改为 `build/conda-debug`

### 构建预设

- `build-conda-release`
- `build-conda-debug`
- `build-test-correctness`
- `build-test-performance`

## 四、README 设计

### 结构

1. 项目简介
2. 功能概览
3. 目录结构
4. 环境要求
5. WSL + Conda 环境配置步骤
6. 使用 `CMakePresets.json` 的配置/编译命令
7. 项目运行说明
8. 正确性测试与性能测试
9. 输入/输出 raw 格式说明
10. 参数文件说明
11. 云服务器兼容说明
12. 常见问题

### 云服务器兼容说明

- 推荐在 Linux 云服务器上复用同一套 Conda 环境创建命令。
- 若云服务器已具备可用 NVIDIA 驱动和 GPU，只需激活 Conda 环境即可复用构建命令。
- `/usr/lib/wsl/lib` 路径仅在 WSL 下生效；在普通 Linux 服务器上缺失时不会要求修改项目源码。

## 五、验收标准

- `cmake --preset conda-release` 可成功生成构建系统。
- `cmake --build --preset build-conda-release` 可成功编译项目。
- `README.md` 能独立指导新用户完成环境配置、编译、运行与测试。
- `README.md` 明确写出 WSL + Conda 主路径，并补充云服务器兼容说明。
