# 集成与复现

仓库：a962695448-rgb/Learning-CUDA，分支：feat/hadamard-cuda。
移植基础提交：cb82ea6e1d5c8f78b48b6922b0f7af279696cc44。

本目录位于 03_hadamard_tc/a962695448-rgb/platforms/moore，使用原生 MUSA API、
独立命名空间和构建入口。共享参考采用现有 include/reference.hpp；服务器测试副本
比仓库原文件多一个末尾空行，代码内容一致，VERIFICATION.json 保留两份字节哈希。

在已安装 MUSA SDK 的 Linux 环境，从项目目录执行：

~~~bash
python3 platforms/moore/run_platform.py --probe-only --output results/moore/probe-new
python3 platforms/moore/run_platform.py --shuffle32 --no-benchmark --output results/moore/validation-new
~~~

输出目录须为新目录。完整运行参数、性能计时范围和 INT4 语义见 README.md；
S4000 实测结果见 REPORT.zh-CN.md，算法推导见 QUANTIZATION_METHOD.md。

本次 GitHub 提交发布已完成实机验证的源码及文档。发布过程中服务器保持关机，
没有因提交操作重复运行 GPU 测试。完整原始文本、计时 CSV 和测试夹具保存在
本地 hadamard-moore-optimized-20260916.zip 交付包中。
