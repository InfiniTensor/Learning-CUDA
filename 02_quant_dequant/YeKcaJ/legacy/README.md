# 历史代码存档

这里保留结构整理前的旧实现、旧构建缓存和兼容入口，便于回看学习过程。**当前程序不编译、不导入这里的代码。**
存档中的脚本、CMake 缓存和文档保留历史路径，不承诺在迁移后直接运行；日常操作只用根目录 README 与 Core。

| 原位置 | 当前位置/用途 |
|---|---|
| CUDAMXFP8、CUDANVFP4、CUDACommon | 本目录同名子目录，旧 CUDA 工程 |
| 根目录 tools、configs | 本目录同名子目录，旧兼容脚本/配置 |
| Core/pipeline.cuh、pipeline_main.cu 等 | previous-Core，拆分前的核心文件 |
| CPUMXFP8、CPUNVFP4 的源码和 golden | 已迁入 **Core/reference/mxfp8、nvfp4**，仍用于正式校验 |
| 两个 CPU 工程的旧 build | build-cache，不再使用；新位置需重新配置构建 |

完整的整理前源码快照：`../records/05-structure-refactor/source-before.tar.gz`。
要复现整理前布局，请把快照解压到另外的空目录，而不是把存档中的旧路径混回正式工程。
这些内容均为迁移保留，没有永久删除历史源码或 golden。
