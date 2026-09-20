# 张量与 packed 文件格式

### 二进制文件协议 v1

所有整数/浮点字段为 little-endian，无隐式结构体 padding。题目仅给出逻辑字段，未指定字符串长度和物理头格式，本实现明确以下协议，并提供 Python 读写工具。

张量头长 32 字节，Python struct 格式 `<8sQQII`：

| 字节偏移 | 类型 | 内容 |
|---:|---|---|
| 0 | char[8] | `LPTENS1\0` |
| 8 | uint64 | rows |
| 16 | uint64 | cols |
| 24 | uint32 | dtype：0=FP32，1=FP16，2=BF16 |
| 28 | uint32 | reserved=0 |
| 32 | dtype[] | 行主序数据，无 padding |

packed 头长 56 字节，Python struct 格式 `<8sQQIIIfQQ`：

| 字节偏移 | 类型 | 内容 |
|---:|---|---|
| 0 | char[8] | `LPPACK1\0` |
| 8 / 16 | uint64 / uint64 | rows / cols |
| 24 | uint32 | format：0=MXFP8，1=NVFP4 |
| 28 | uint32 | block_size |
| 32 | uint32 | tensor scaling：0/1 |
| 36 | float32 | global scale；MXFP8 固定 1 |
| 40 / 48 | uint64 / uint64 | scale 字节数 / packed data 字节数 |
| 56 | uint8[] | scale 数组，随后是 packed data |

BF16 以 16 位原始位模式存储，使用整数实现 RNE 转换，在 `sm_75` 上可运行。文件格式为线性软件交换格式，不是 Blackwell GEMM 的 swizzled 硬件布局。
