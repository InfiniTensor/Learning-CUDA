# 结构整理验证（2026-09-11）

本次拆分模块、迁移目录和更新构建/文档，不作为新一轮算子性能优化。

| 检查 | 结果 | 证据 |
|---|---|---|
| kernel、设备编码、scale 定义 | 15 个定义搬迁前后逐字节一致 | kernel-preservation.json |
| CPU 源码 | 两份 main.cpp 与整理前快照逐字节一致 | source-before.tar.gz 与 Core/reference 下源码 |
| 正式构建与回归 | Release 构建成功，7/7 CTest 通过 | build.log、ctest.log、LastTest.log |
| 冻结输入/golden | 两种格式各 15 个 SHA256 全部通过 | *-frozen-hashes.log |
| Core 独立构建 | 仅复制 Core 并改名 standalone，BUILD_TESTING=OFF 构建两个后端成功 | standalone-configure.log、standalone-build.log |
| 独立构建文件测试 | 每种格式各 11 项通过，含 FP32/FP16 输入、模式/舍入和三种输出类型 | *-standalone-io.log |
| 自测默认路径 | 无需手工传 CPU 目录，两种格式均通过 | *-default-selftest.log |
| benchmark 入口 | 两种格式 4M、5 次重复成功；用于检查入口，不发布新的加速比 | *-benchmark-smoke.jsonl |
| nsys | 两种格式均采到对应 fused kernel、CUDA API 与内存传输 | profile/*_stats.txt |

整理前源码快照 SHA256：`6ee4f91fea590b6e42775c60fb943dcc91718e6bc929cd60a7277804c73b2125`。
旧源码与旧构建缓存迁入 legacy；已有 input、results、前四阶段 records 及优化日志保留原位置。
本轮未重新运行 Compute Sanitizer，不沿用旧日志声明本轮检查通过。
