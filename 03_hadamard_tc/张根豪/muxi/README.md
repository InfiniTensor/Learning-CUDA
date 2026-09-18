# 沐曦适配

此目录是 FastHadamard-CUDA 的独立沐曦构建入口，源码基于仓库 CUDA 实现复制，避免影响 CUDA 与 `musa/` 目录。

## 环境

当前环境已安装 MACA 3.5.3，默认使用 `/opt/maca-3.5.3/mxgpu_llvm/bin/mxcc`，CUDA 兼容头位于 `tools/cu-bridge/include`。若实际 SDK 路径或编译器名称不同，请覆盖变量：

```bash
cd 03_hadamard_tc/张根豪/muxi
source env.sh
make MUXI_HOME=/path/to/maca MXCC=/path/to/mxcc
```

也可以通过 `MUXI_EXTRA_FLAGS` 传入 SDK 版本对应的架构或兼容选项，例如：

```bash
make MUXI_ARCH=<沐曦架构> MUXI_EXTRA_FLAGS="<SDK额外编译参数>"
```

## 构建与验证

```bash
make
make smoke
make clean
```

适配代码保留 CUDA 兼容头（`cuda_fp16.h`、`cuda_bf16.h`）和 CUDA 风格内建函数，适用于提供 CUDA 兼容编程接口的沐曦编译器；若其他 SDK 版本要求专用头文件或内建函数名，请在 `src/` 中建立对应兼容层后再构建。
