# 数值规则与约定（实现前确定，2026-09-20）

依据：[OCP MX v1.0，2023-09](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) §5–6；[NVIDIA NVFP4，2025-06-24](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)。前者确定元素和 E8M0 表示，后者确定 NVFP4 的 E2M1、16 元素 E4M3 scale 和 FP32 全局 scale。以下 scale 选择、文件容器及随机算法是本项目自行确定的策略。

|元素|符号/指数/尾数|bias|最小次正规/正规|最大有限|特殊编码|
|---|---|---|---|---|---|
|E4M3（OCP finite，非 FNUZ）|1/4/3|7|2^-9 / 2^-6|448|0x7f/ff NaN，无 Inf，±0|
|E5M2|1/5/2|15|2^-16 / 2^-14|57344|0x7c/fc Inf，指数全一且尾数非零 NaN，±0|
|E2M1|1/2/1|1|0.5 / 1|6|无 NaN/Inf，±0|

正规值 (-1)^s 2^(e-bias)(1+m/2^M)，次正规值 (-1)^s 2^(1-bias)m/2^M。
元素转换采用有限饱和；FP8 NaN 规范化为正 canonical NaN，E2M1 NaN 映射 +0；Inf 饱和保留符号。矩阵输入拒绝 NaN/Inf。底层解码仍保留上述特殊编码。当前仅采用 SAT，未实现 OCP 全部可选/必需转换操作（OVF 尚未实现）。

MX：块最大绝对值 a，选择 e=clamp(ceil(log2(a/max_element)),-127,127)，存储 E8M0 字节 e+127；零块存 127（scale=1）。实现使用 double frexp 和精确比较，避免 log2 临界误判。字节 0..254 表示 2^(byte-127)，255 为 NaN，容器拒绝 NaN scale。q=encode(x/decoded_scale)，y=decode(q)*decoded_scale。

NV：A 为张量最大值，g=FP32(max(A/(6*448),2^-126))；全零 g=1。局部 s=E4M3_RNE(a/(6*g))，非零 a 的 s 若舍入为零则提升至最小正次正规；零块 s=1。q=E2M1(x/(g*decode(s)))，y=decode(q)*decode(s)*g。g 是反量化乘数，不是其倒数。使用编码后 scale 量化。中间计算使用 double 防止 scale 乘积下溢；最终 FP32 溢出饱和至最大有限值。

nearest 为相邻值距离比较，精确中点选编码最低位偶数。stochastic 使用距离概率 p=(abs(x)-lo)/(hi-lo)，u < p 选 hi；负数在绝对值域对称处理。u 来自 SplitMix64(seed XOR (index+0x9e3779b97f4a7c15)) 的高 53 位，索引为行主序逻辑元素，与线程布局无关。饱和区无随机化，可表示值不改变，±0 保留。随机概率受 53 位均匀网格离散化。scale 永远确定性 RNE（MX 向上取幂）；最终 FP16/BF16 转换独立使用 RNE，溢出为 Inf。

block 模式沿张量的行方向分块：每行独立按 block_size 切块，块不跨行。块号 b = r * bpr + c / block_size，其中 r = i / cols，c = i % cols，bpr = ceil(cols / block_size)。每行最后一个块可能不满，无元素 padding。默认 MX=32、NV=16。tensor 模式只存一个局部 scale，属于扩展。非默认块大小不支持，明确报错。非完整尾块属于容器扩展，元数据标记。NV 偶数索引在低半字节、奇数索引在高半字节，奇数总长度的最后高半字节填零；行尾不 padding。

本项目模拟数值和紧凑存储，不是 Blackwell MMA scale swizzle 布局，不使用原生 FP8/FP4 指令。其他架构需另行编译和验证。
