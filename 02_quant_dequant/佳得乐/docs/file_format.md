# 文件格式 v1

所有整数与 IEEE 浮点位均为 little endian。当前机器读写 payload 使用本机 little endian（启动时检查），header 逐字段写，不写 C++ struct。shape 字段为正 int64；实现拒绝非正、乘法溢出及 payload 不匹配。dtype: 0=fp32,1=fp16,2=bf16（仅输出）。

## 张量容器

|offset|宽度|字段|
|---|---|---|
|0|8|ASCII LPTENSOR|
|8|4|version=1|
|12|4|endian=0x01020304|
|16|8|rows|
|24|8|cols|
|32|4|dtype|
|36|4|payload offset=48|
|40|8|payload bytes|
|48|N*元素字节|行主序 payload|

## 低精度容器

|offset|宽度|字段|
|---|---|---|
|0|8|ASCII LPPACKED|
|8|4|version=1|
|12|4|endian tag|
|16,24|各8|rows,cols|
|32|4|input dtype|
|36|4|format:0=MX-E4M3,1=MX-E5M2,2=NV-E2M1（含 fp8_type）|
|40|4|block_size（32/16）|
|44|4|scale_mode:0=block,1=tensor|
|48|4|output dtype|
|52|4|rounding:0=nearest,1=stochastic|
|56|8|seed|
|64|4|scale type:0=E8M0,1=E4M3|
|68|4|extension flags:bit0=tensor,bit1=block 模式下每行最后一个块不满|
|72|4|FP32 global dequant scale，MX 固定1|
|76|4|header bytes=112|
|80|8|scale count S|
|88|8|scale offset=112|
|96|8|packed bytes P|
|104|8|packed offset=112+S|
|112|S|scale bytes|
|112+S|P|packed payload|

总长度严格为112+S+P，无附加字节。block 模式下 S 为 rows * ceil(cols/block_size)，tensor 模式为1。MX P=N；NV P=ceil(N/2)。NV 元素 2j 在 byte[j] 低4位，2j+1 在高4位。奇数 N 最后高4位必须0。

量化块沿张量的行方向划分：每行独立按 block_size 切块，块不跨行。每行最后一个块可能不满，无元素 padding，不做行尾 padding。

读取验证 magic/version/endian/shape/dtype/format/全部枚举/scale类型/扩展标识/偏移/长度/scale有效性/奇数 padding。拒绝 E8M0 NaN、NV 零/负/NaN scale，以及非正/非有限全局 scale。低精度元素允许格式原生特殊编码；矩阵量化输入拒绝非有限数。

target_gpu 仅在配置和实验元数据记录，不决定解码。独立 dequantize 仅需低精度文件。没有 checksum，因此合法字节翻转不保证被发现；结构损坏会拒绝。
