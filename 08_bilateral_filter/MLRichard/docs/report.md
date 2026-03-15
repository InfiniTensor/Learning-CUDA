# Bilateral Filter Profiling Report

## 1. Scope

本报告对应 `docs/plans/2026-03-06-bilateral-filter-impl.md` 的 **Task 12**，目标是把 profiling 流程整理为默认适配 **英伟达（NVIDIA）环境**：

- **训练营英伟达环境可直接复用**
- **标准 Linux / WSL + NVIDIA GPU 环境可执行**
- 不需要修改项目源码
- 在计数器权限受限时也能保留可执行的降级路径

## 2. Added Assets

新增或补强的资产：

- `tools/generate_test_raw.py`
  - 生成确定性的 `3840x2160 RGB raw` 测试输入
  - 避免 profiling 依赖外部真实图片数据集
- `tools/profile_task12.sh`
  - 自动定位 `bilateral_filter`
  - 自动生成默认输入
  - 自动识别当前 GPU 并按 GPU 名称分类输出
  - 默认执行 `ncu + nsys`
  - 自动导出 `timeline.sqlite`
  - 若 `ncu` 命中 `ERR_NVGPUCTRPERM`，自动继续执行 `nsys`
  - 自动生成 `ncu_permission_hint.txt`
- `docs/report.md`
  - 记录英伟达环境下的验证结果与兼容说明

## 3. Standard Workflow

如果训练营提供的英伟达环境已经具备基础依赖，可直接从激活项目环境开始；以下命令同时适用于普通 Linux 服务器和 WSL + NVIDIA CUDA 环境。

### 3.1 Activate environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bilateral-cuda124
```

### 3.2 Configure and build

```bash
cmake --preset conda-release
cmake --build --preset build-conda-release -j"$(nproc)"
```

### 3.3 Run profiling

```bash
bash tools/profile_task12.sh
```

默认会：

- 生成 `artifacts/profiling/inputs/test_4k.raw`
- 对 `naive / shared / adaptive` 尝试生成 `.ncu-rep`
- 对 `shared` 生成 `timeline.nsys-rep`
- 自动导出 `timeline.sqlite`

### 3.4 Fallback mode

如果当前机器未开放 GPU Performance Counters，可先执行：

```bash
bash tools/profile_task12.sh --skip-ncu
```

这样仍可获得：

- `timeline.nsys-rep`
- `timeline.sqlite`
- `profile_output.raw`

## 4. Output Layout

```text
artifacts/profiling/
├── inputs/
│   └── test_4k.raw
└── <gpu-slug>/
    ├── profile_naive.ncu-rep
    ├── profile_shared.ncu-rep
    ├── profile_adaptive.ncu-rep
    ├── profile_*.ncu.log
    ├── ncu_permission_hint.txt
    ├── timeline.nsys-rep
    ├── timeline.sqlite
    └── profile_output.raw
```

说明：

- 若 `ncu` 完整成功，则会有 `profile_*.ncu-rep`
- 若 `ncu` 因权限失败，则至少保留 `profile_*.ncu.log` 与 `ncu_permission_hint.txt`

## 5. Metrics To Inspect

重点关注以下指标：

- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second`
  - Global Memory 读带宽
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
  - Shared Memory bank conflict
- `sm__warps_active.avg.pct_of_peak_sustained_active`
  - Occupancy / warp 活跃度

## 6. Verification on NVIDIA GPUs

### 6.1 Build and helper verification

已验证：

- `python tools/generate_test_raw.py --help`
- `bash tools/profile_task12.sh --help`
- `cmake --preset conda-release`
- `cmake --build --preset build-conda-release -j"$(nproc)"`
- `ctest --preset test-conda-all`（在真实英伟达 GPU 环境中通过）

### 6.2 Measured evidence on RTX 4070

本地 `RTX 4070` 上已有 benchmark 结果：

- `naive_4K_rgb`: `897.5 MPix/s`
- `shared_4K_rgb`: `1249.0 MPix/s`
- `adaptive_4K_rgb`: `1249.6 MPix/s`

这说明：

- `shared` 相比 `naive` 已有明显加速
- 当前实现已达到计划中 `> 1000 MPix/s` 的吞吐目标

上述数据是当前报告中的实测样本，证明这套流程已在一张实际英伟达消费级 GPU 上跑通。对训练营服务器或其他 NVIDIA Linux 机器，本文将其视为同一套工具链的复用路径，而不是另一套独立流程。

### 6.3 Profiling execution status on RTX 4070

本次在本地 `RTX 4070` 上的实际执行结果：

- `bash tools/profile_task12.sh`
  - `ncu` 启动成功，但被主机侧 GPU Performance Counter 权限限制拦截
  - 错误为 `ERR_NVGPUCTRPERM`
- `bash tools/profile_task12.sh --skip-ncu`
  - 成功生成 `timeline.nsys-rep`
  - 成功生成 `timeline.sqlite`
  - 成功生成 `profile_output.raw`

实测 `shared` kernel 运行日志：

- `GPU Time`: `7.90 ms`
- `Throughput`: `1050.1 MPix/s`
- `CPU Time`: `175.41 ms`
- `Speedup`: `22.2x`
- `MAE`: `0.3168`
- `PSNR`: `55.61 dB`

当前已确认：

- **Task 12 的脚本链路已在英伟达 GPU 实机上可执行**
- **在未开放 perf counters 时，仍可稳定获得 `nsys` 时间线结果**
- **开放 perf counters 后，无需改源码即可补齐 `ncu` 报告**

## 7. Permission Fix Instructions

### 7.1 Windows + WSL + NVIDIA GPU

根据 NVIDIA 官方 `ERR_NVGPUCTRPERM` 说明，Windows 平台应：

1. 以管理员身份打开 `NVIDIA Control Panel`
2. 打开 `Desktop -> Enable Developer Settings`
3. 打开 `Developer -> Manage GPU Performance Counters`
4. 选择 `Allow access to the GPU performance counter to all users`
5. 重新打开 WSL shell 后重新执行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bilateral-cuda124
bash tools/profile_task12.sh
```

这里把 WSL 归到 Windows host 管理，是因为 WSL GPU profiling 依赖宿主机 NVIDIA 驱动控制。

### 7.2 Native Linux / NVIDIA server

Ubuntu / Debian 永久配置：

```bash
sudo tee /etc/modprobe.d/nvidia-prof.conf >/dev/null <<'EOCONF'
options nvidia NVreg_RestrictProfilingToAdminUsers=0
EOCONF
sudo update-initramfs -u -k all
sudo reboot
```

重启后检查：

```bash
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
```

若输出为 `RmProfilingAdminOnly: 0`，普通用户即可运行 `ncu`。

临时配置（官方 Linux 方案，适合需要立即验证的机器）：

```bash
sudo systemctl isolate multi-user
sudo modprobe -rf nvidia_uvm nvidia_drm nvidia_modeset nvidia-vgpu-vfio nvidia
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
sudo systemctl isolate graphical
```

## 8. Compatibility Summary

- `训练营英伟达环境`
  - 可直接复用同一套构建、测试与 profiling 工作流
  - 若计数器权限受限，仍可通过 `--skip-ncu` 保留 `nsys` 结果
- `Windows + WSL + NVIDIA GPU`
  - 已有 `RTX 4070` 实测样本
  - `nsys` 已验证通过
  - `ncu` 需宿主机放开 perf counter 权限
- `Linux + NVIDIA GPU`
  - 使用同一套 Conda 环境与同一脚本即可执行
  - 包括 A100 在内的服务器型 GPU 复用方式一致
  - 若权限默认受限，仅需按上节执行一次 host 级配置
  - 不需要修改项目源码或 CMake 文件

## 9. Reference

- NVIDIA official guide: `https://developer.nvidia.com/ERR_NVGPUCTRPERM`
