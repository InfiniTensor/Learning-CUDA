# v0 baseline

冻结日期：2026-09-20。

源码 `luxorion_cli.cu` 是正式功能闭环在进入八项优化前的快照。支持MXFP8/NVFP4、FP32/原生FP16输入、tensor/block scaling、nearest/stochastic和FP16/BF16/FP32输出。

本目录源码SHA256与冻结时正式入口一致。性能基线使用warmup=10、repeats=50并连续运行3轮。
