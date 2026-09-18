# `src/formats` 编码器原理与源码导读

`src/formats/` 是项目的**单元素数值编码层**。它不负责遍历矩阵、分配 GPU
内存、计算 CUDA grid，也不负责 QDWGT 文件读写；它只回答三个问题：

1. 一个 FP32 数如何编码为 E4M3 或 E2M1 的 bit pattern；
2. 一个低精度 bit pattern 如何恢复为 FP32；
3. scale 如何编码、解码，以及如何与元素编码组合。

这三个头文件都是 header-only，并把关键函数标为 `__host__ __device__`。因此
CPU reference 和 CUDA kernel 调用的是同一套规则，才能逐字节比较 payload 与 scale。

| 源文件 | 角色 | 主要内容 |
| --- | --- | --- |
| `fp32_utils.cuh` | 公共基础 | FP32 按位读取、指数分析、精确 2 的幂、RNE 辅助与随机数收窄。 |
| `mxfp8_codec.cuh` | MXFP8 codec | E4M3 元素、E8M0 scale、MXFP8 单元素量化/反量化、SplitMix64 随机数。 |
| `nvfp4_codec.cuh` | NVFP4 codec | E2M1 nibble、双层 scale、NVFP4 单元素量化/反量化、两个 nibble 的打包。 |

数值格式的项目级规范、block 布局和 I/O 约定见
[format_spec.md](format_spec.md)；本文着重解释源码中每条数值路径为何成立。

## 读 codec 前需要的三个概念

### 位字段、bias 与隐含 1

浮点编码通常把一个值拆成符号位、指数位和 fraction（尾数存储位）。以一般形式的
`S E...E M...M` 为例，`S` 决定正负，`E` 是编码后的指数，`M` 是 fraction。

- **biased exponent（偏置指数）**：指数位是无符号整数，不能直接存负指数。因此保存
  $E=e+\mathrm{bias}$；$e$ 才是数学上的 unbiased exponent（真实指数）。
- **normal（规格化数）**：当指数位不为零，二进制有效数在 $[1,2)$，最高的 `1` 对所有
  normal 数都相同，因此不存它。数值形如 $(-1)^S(1+M/2^m)2^e$。
- **subnormal（非规格化数）**：指数位为零时，隐含的 `1` 被取消，数值从零开始均匀排布。
  这样能填补零和最小 normal 之间的空隙，但精度低于 normal 数。

这里的 `M` 既可能指“存储的无符号整数”，也可能泛指 fraction field；代码会用
`mantissa` 表示它。**magnitude code（幅值码）** 则是去掉符号位后剩下的所有 bit：
它只描述非负幅值，最后再拼回正负号。

### 为什么先缩放再编码

低精度 FP 格式不是 INT quantization 的 `round(x / s) + z`。它仍然保留指数和
fraction，元素编码器本身就是一张非均匀的浮点数表。scale 的作用是把一个 block 的
动态范围移动到该表能容纳的范围：

$$
z_i = x_i / s, \qquad p_i = Q_{\mathrm{lowp}}(z_i), \qquad
\hat{x}_i = \operatorname{decode}(p_i)\,s.
$$

这里 $z_i$ 是缩放后的 FP32 临时值，$p_i$ 是真正写入 payload 的低精度 code；两者
不是同一种东西。FP8/FP4 的量化依旧需要 scale，是因为单个低精度浮点数的指数范围远小于
FP32，且有限个 fraction bit 决定了相对精度。

### 两种舍入目标不同

- `nearest` 调用 RNE（round-to-nearest, ties-to-even）：选择距离最近的可表示值；正好
  在中点时选择最低有效 bit 为偶数的 code。它通常降低单次误差。
- `stochastic` 先找相邻下界 $l$ 和上界 $h$，再以
  $P(h)=(|x|-l)/(h-l)$ 的概率选上界。它的期望值等于原值，长期统计偏差较小，但某一
  次实验的 MAE/MSE 往往大于 RNE。

两个模式对精确可表示值、零和已超出最大值的饱和输入都不需要随机数。

## `fp32_utils.cuh`：让 host 与 device 以同一方式理解 FP32

### 重要接口：输入、输出与用途

| 接口 | 输入 | 输出 | 用途 |
| --- | --- | --- | --- |
| `float_to_bits(value)` | 一个 FP32 `value` | 同 bit pattern 的 `uint32_t` | 无损读取 FP32 的符号、指数和 fraction。 |
| `bits_to_float(bits)` | 一个 IEEE FP32 bit pattern | 同 bit pattern 的 FP32 | 从构造出的 bit 恢复 FP32，例如固定 NaN、`-0` 和 $2^{-127}$。 |
| `floor_log2_positive(x)` | 正且有限的 FP32 | 整数 $\lfloor\log_2(x)\rfloor$ | 找 normal 编码所需的真实指数。 |
| `power_of_two(e)` | 整数指数 `e` | 精确的 FP32 $2^e$ | 让 codec 不依赖近似的 `powf`，并正确覆盖 $2^{-127}$。 |
| `round_to_nearest_even_nonnegative(x)` | 范围受调用方限制的非负有限 FP32 | RNE 后的 `uint32_t` | 对 subnormal 单位或 mantissa 格子做整数舍入。 |
| `clamp_uniform_random(u)` | 候选随机 FP32 | $[0,1)$ 内的 FP32 | 为 stochastic rounding 修正 NaN、负数和 `1.0F`。 |

### FP32 的真实 bit 布局

IEEE 754 binary32 的 32 bit 为 `S EEEEEEEE M...M`：1 个符号位、8 个指数位、23 个
fraction 位，bias 为 127。代码以 `uint32_t` 掩码读取它们：

| 常量 | bit pattern | 含义 |
| --- | --- | --- |
| `kSignMask` | `0x80000000` | bit 31；也能区分 `+0` 和 `-0`。 |
| `kExponentMask` | `0x7f800000` | bits `[30:23]`；全 1 表示 NaN 或 Inf。 |
| `kFractionMask` | `0x007fffff` | bits `[22:0]`。 |
| `kCanonicalQuietNaNBits` | `0x7fc00000` | 项目固定的 quiet NaN，保证 CPU/GPU 一致。 |

`float_to_bits()` 和 `bits_to_float()` 是**按位重解释**，不是数值转换：在 device 使用
CUDA intrinsic，在 host 用 `memcpy` 避免 strict-aliasing 未定义行为。因此 NaN payload
和 `-0` 的符号都不会丢失。

`is_finite()` 不调用不同平台可能实现不同的数学库，而是直接检查指数是否全 1。
`absolute_value()` 只是清除符号位；这比调用 `fabsf` 更明确地保留 NaN 的其他 bit。

### 为什么 `floor_log2_positive()` 要特别处理 subnormal

normal FP32 的真实指数就是 `exponent_field - 127`。但 subnormal 的 exponent field
为零，不能机械地得到 $-127$：它的值是
$\mathrm{fraction}\times2^{-149}$。因此函数找到 fraction 的最高置位 bit：若该 bit
位于 `k`，则 $\lfloor\log_2(x)\rfloor=k-149$。

例如最小 FP32 subnormal 的 fraction 是 1，因此得到 $0-149=-149$；值为
$2^{-127}$ 的 fraction 最高位是 22，因此得到 $22-149=-127$。

`power_of_two(exponent)` 是反方向的精确构造。普通 $2^e$ 可直接写 exponent field，
但 $2^{-127}$ 是 FP32 subnormal，必须特别构造 `0x00400000`；这正是 E8M0 最小
scale 解码时需要的边界。

### RNE 和随机数的公共辅助函数

`round_to_nearest_even_nonnegative()` 将一个小的非负数拆成整数部分和小数部分：

- 小数大于 `0.5`，向上；
- 小数小于 `0.5`，向下；
- 恰好为 `0.5`，仅当截断结果是奇数时向上，使结果为偶数。

它用于“以最小 subnormal 为单位”的整数舍入和 mantissa 舍入，不是任意范围 FP32 的
通用整数转换器。

`clamp_uniform_random()` 则把调用者给出的候选随机数收窄至 $[0,1)$。`1.0F` 会映射为
`kLargestBelowOne = 1-2^{-24}`，所以 stochastic rounding 不会意外以概率 1 选择上界。

## `mxfp8_codec.cuh`：E4M3、E8M0 与 MXFP8

### 重要接口：输入、输出与用途

| 接口 | 输入 | 输出 | 用途 |
| --- | --- | --- | --- |
| `decode_e4m3(encoded)` | 完整 E4M3 byte `S EEEE MMM` | FP32 | 解释一项 MXFP8 payload。 |
| `find_e4m3_lower_magnitude_code(magnitude)` | 有限 $[0,448)$ 的非负 FP32 幅值 | 不含符号的 7-bit `EEEE MMM` code | 找到不超过输入的下界，供 RNE/SR 选择相邻端点。 |
| `encode_e4m3(value, rounding, u)` | FP32、舍入模式、SR 随机数 | 完整 E4M3 byte | 将缩放后的元素编码，并处理 NaN、零和饱和。 |
| `decode_e8m0(encoded)` | 一个 E8M0 scale byte | 正 FP32 scale 或 NaN | 将 scale code 解释为 $2^{e-127}$。 |
| `encode_e8m0_round_up(scale)` | 非负 FP32 所需 scale | E8M0 scale byte | 向上取 2 的幂，避免 MXFP8 最大元素溢出。 |
| `compute_mxfp8_scale_code(amax)` | 一个 block 或整张张量的非负 `amax` | E8M0 scale byte | 把范围 $amax$ 转为可直接使用的 MXFP8 scale。 |
| `encode_mxfp8_element(value, scale, rounding, u)` | 原始 FP32 元素、E8M0 scale byte | E4M3 payload byte | 组合“除 scale + E4M3 编码”。 |
| `decode_mxfp8_element(payload, scale)` | E4M3 payload、E8M0 scale byte | FP32 | 组合“解码 E4M3 + 乘 scale”。 |

### E4M3 元素表

E4M3 payload 是完整的一个 byte，布局为 `S EEEE MMM`，指数 bias 为 7。

| 条件 | 解码公式 | 关键点 |
| --- | --- | --- |
| `E = 0, M = 0` | 带符号零 | decoder 保留输入文件中的 `-0`。 |
| `E = 0, M > 0` | $(-1)^S M \times 2^{-9}$ | subnormal，间距固定为 $2^{-9}$。 |
| `1 ≤ E ≤ 15` 且不是 NaN | $(-1)^S(1+M/8)2^{E-7}$ | normal，具有隐含 `1`。 |
| `E = 15, M = 7` | NaN | `0x7f` 与 `0xff` 均是 NaN。 |

E4M3 不保留 Inf：`E=15,M=6` 是最大有限值 `448`，code 为 `0x7e`；再大的有限值和
`±Inf` 都饱和到同符号的 `±448`。最小 normal 是 $2^{-6}$，最小 subnormal 是
$2^{-9}$。后者来自 $M=1$ 时的 $1\times2^{-9}$，而不是 bias 直接计算出的值。

### `decode_e4m3()`：bit field 到 FP32

该函数先拆出 `negative`、`exponent_field` 和 `mantissa`：

1. 若 `E=15,M=7`，返回固定的 `fp32::canonical_quiet_nan()`；FP32 有确定的 NaN
   表示，后续浮点运算也能正常传播它。
2. 若 `E=0,M=0`，用 `signed_zero(negative)` 恢复 `+0` 或 `-0`。
3. 若 `E=0`，计算 `M * kE4M3MinSubnormal`。
4. 否则计算 `(8 + M) * 2^(E - 10)`。它与
   $(1+M/8)2^{E-7}$ 完全等价，只是整数形式适合代码。

### `find_e4m3_lower_magnitude_code()`：为何先找下界

**输入**是一个已取绝对值的 FP32 `magnitude`；调用方保证它有限且落在 $[0,448)$。
**输出**是一个不带符号位的 `uint8_t`，范围为 `0x00` 到 `0x7e`。将它传给
`decode_e4m3()` 后，得到的值一定不大于输入幅值，且它是满足该条件的最大 E4M3 值。

**用途**不是直接完成量化，而是先定位相邻端点。RNE 需要该下界和下一 code 的中点；
stochastic rounding 需要它们之间的距离来计算选上界概率。调用者最后才把原输入的符号位
拼到这个 magnitude code 上。

编码器需要知道输入夹在哪两个相邻值之间。例如 1.30 的相邻 E4M3 值是 1.25 和 1.375。
RNE 要比较它们的中点；stochastic 要计算选上界的概率。因此 helper 返回**不超过输入的
最大幅值码**，再由 `lower_code + 1` 获得上界。

- 若幅值小于 $2^{-6}$，它在 subnormal 区：最小步长为 $2^{-9}$，所以
  `magnitude * 512` 就是“有多少个 subnormal 单位”。`float -> uint32_t` 对非负数截断，
  等价于向下取整，得到最大的 $M$ 使 $M \times 2^{-9}\leq |x|$。
- 否则先用 `floor_log2_positive()` 得到真实指数 $e$，然后算
  `normalized = magnitude / 2^e`。它一定在 $[1,2)$；`(normalized - 1) * 8` 是
  三个 mantissa bit 对应的 8 个区间编号，截断后得到下界 `M`。
- 最后把 `E=e+7` 左移 3 bit，并与 `M` 按位或，得到不含符号的 `EEEE MMM`。

这个函数只接收 $[0,448)$ 的有限非负数，所以不会返回 NaN code `0x7f`。

### `encode_e4m3_rne_sat()`：RNE 编码完整流程

1. 先从原 FP32 取符号位；对 NaN 返回 canonical `0x7f`，对 Inf 返回同符号的最大有限
   code。
2. 取幅值。`+0` 和 `-0` 都写为 `0x00`，使项目自己的量化输出只有一种零的 bit
   pattern；decoder 仍兼容外部文件中的 `0x80`。
3. 幅值不小于 448 时饱和到 `0x7e` 或 `0xfe`。
4. 对 subnormal 区，计算 `magnitude * 512` 并 RNE 到整数单位。结果 8 会自然变成
   `E=1,M=0`，即最小 normal，不会错误留在 subnormal。
5. 对 normal 区，将 $[1,2)$ 的 fraction 乘 8 并 RNE。若 mantissa 从 7 进位为 8，
   清零 mantissa 并给真实指数加一。
6. 最后检查指数进位不能落入 E4M3 的 NaN 码 `E=15,M=7`；这种有限溢出仍饱和在
   `E=15,M=6`。

### `encode_e4m3_stochastic_sat()`：与 RNE 的差别

前面的 NaN、零和饱和处理相同；它只改变“严格夹在两个有限 codebook 值之间”的情况。
假设下界为 $l$、上界为 $h$，则计算
$p=(|x|-l)/(h-l)$。随机数小于 $p$ 选择上界，否则选择下界。因此
$\mathbb{E}[Q(x)]=x$；符号位最后原样拼回。

`encode_e4m3()` 只是根据 `RoundingMode` 分派到上面两个函数；`kUnknown` 的安全回退为
RNE，但上层配置校验本来就应拒绝未知模式。

### E8M0：为什么 scale 只能是 2 的幂

E8M0 的 byte 不含符号和 mantissa，只有 8 bit 编码指数。有限 code $e\in[0,254]$ 解码为
$2^{e-127}$，`0xff` 是 NaN；它没有零也没有 Inf。

这意味着 scale 只能是 2 的整数次幂。好处是它的表示不额外引入 mantissa 误差，而且
乘除 scale 在二进制浮点中通常只是指数移动；代价是范围会向上取整，不能恰好贴住
block 的 `amax`。

`encode_e8m0_round_up(required_scale)` 的目标不是“最接近”，而是
$\lceil\log_2(\mathrm{required\_scale})\rceil$。它先使用 FP32 bit 分析判断输入是否
恰是 2 的幂：是则不加一，不是则在 `floor_log2` 后加一；再加 bias 127 并夹在 `[0,254]`。
向上取整保证 scale 不会太小。

`compute_mxfp8_scale_code(amax)` 把这个规则接到格式范围上：

$$
s=\operatorname{E8M0\_UP}(amax / 448).
$$

于是 $amax/s\leq448$，最大元素不会在 E4M3 编码时因 scale 偏小而溢出。`amax=0` 时
写 `0x00`，它表示最小有限 scale $2^{-127}$；不是数学零，但零 payload 乘它后仍为零。

### MXFP8 单元素组合接口

`encode_mxfp8_element(value, scale_code, rounding, u)` 先 `decode_e8m0(scale_code)` 得到
scale，再编码 $z=value/s$ 为 E4M3。若 scale code 是 E8M0 NaN，则返回 E4M3 canonical
NaN，避免继续做无意义除法。

`decode_mxfp8_element(payload, scale_code)` 正好反向执行：

$$
\hat{x}=\operatorname{decode\_e4m3}(payload)
\times\operatorname{decode\_e8m0}(scale\_code).
$$

在 block mode 中上层为每个 rowwise 32 元素 block 传不同的 `scale_code`；在 tensor mode
中全部元素传同一个 code。codec 本身不关心 scale 的索引来自哪里。

### 可复现的 stochastic 随机数

`mxfp8_stochastic_uniform_for_element(seed, linear_index)` 使用无状态 SplitMix64：
输入只由配置 seed 和 row-major 全局下标决定，再取混合结果高 24 bit 乘 $2^{-24}$。

因此 CPU 即便按顺序循环、GPU 即便以 persistent grid-stride 顺序处理，两边同一个元素都
得到同一个 $u\in[0,1)$。NVFP4 的
`nvfp4_stochastic_uniform_for_element()` 直接复用这个映射，保证两个格式的随机规则一致。

## `nvfp4_codec.cuh`：E2M1 与双层缩放

### 重要接口：输入、输出与用途

| 接口 | 输入 | 输出 | 用途 |
| --- | --- | --- | --- |
| `decode_e2m1(encoded)` | 低 4 bit 的 E2M1 nibble | FP32 | 解码一个 NVFP4 payload 元素。 |
| `find_e2m1_lower_magnitude_code(magnitude)` | 有限 $[0,6)$ 的非负 FP32 幅值 | 不含符号的 3-bit `EE M` code | 找 E2M1 下界，供两种舍入复用。 |
| `encode_e2m1(value, rounding, u)` | 已缩放 FP32、舍入模式、SR 随机数 | 高 4 bit 为零的 E2M1 nibble | 对单元素进行 FP4 数值编码。 |
| `compute_nvfp4_global_scale(tensor_amax)` | 整张张量的非负 `amax` | FP32 global decode scale | 建立全张量范围的粗粒度缩放。 |
| `compute_nvfp4_local_scale_code(block_amax, global_scale)` | 16 元素 block 的 `amax`、global scale | E4M3 local-scale byte | 让该 block 的 E2M1 范围尽量用满。 |
| `encode_nvfp4_element(value, local, global, rounding, u)` | 原元素、E4M3 local code、FP32 global scale | E2M1 nibble | 组合两层缩放与 E2M1 编码。 |
| `decode_nvfp4_element(payload, local, global)` | E2M1 nibble、E4M3 local code、FP32 global scale | FP32 | 组合三项，得到反量化结果。 |
| `pack_e2m1_nibbles(low, high)` | 两个 E2M1 nibble | 一个 `uint8_t` | 将两个 4-bit 元素真正压缩为一个字节。 |

### E2M1 nibble 的完整 codebook

一个 E2M1 payload 是 4 bit `S EE M`，其中 `S` 是 bit 3，`EE` 是两位指数，`M` 是一位
mantissa，bias 为 1。去掉符号后的所有八种 code 都是有限值：

| 幅值码 `EE M` | 数值 | 分类 |
| --- | ---: | --- |
| `000` | 0 | 零 |
| `001` | 0.5 | subnormal |
| `010` | 1 | 最小 normal |
| `011` | 1.5 | normal |
| `100` | 2 | normal |
| `101` | 3 | normal |
| `110` | 4 | normal |
| `111` | 6 | 最大有限值 |

当 `EE=0`，没有隐含 1，`M=1` 得到 $0.5$；当 `EE>0`，公式为
$(1+M/2)2^{EE-1}$。E2M1 没有 NaN/Inf 保留码，所以完整量化 pipeline 在进入 codec
之前拒绝原始 NaN/Inf。

`decode_e2m1()` 先以 `decode_e2m1_magnitude()` 查上述 codebook，再恢复符号；它保留
外部 nibble 中的 `-0`。`find_e2m1_lower_magnitude_code()` 直接按表中边界比较。这里只有
8 个值，查表式分支比套用通用指数公式更清晰，也能准确处理 0.5 到 1 的边界。

### E2M1 的 RNE 与 stochastic 编码

`encode_e2m1_rne_sat()` 与 E4M3 的总体策略相同，但相邻间距不再规则：

- 非零有限幅值大于等于 6 时饱和为同符号 `±6`；
- 在下界/上界间算中点；正好中点时检查 `lower_code & 1` 来实现 ties-to-even；
- NaN 在这个孤立的单元素接口中规范化为正零，因为 E2M1 没有 NaN code；完整 pipeline
  会更早拒绝它；
- `-0` 写为正零，保持项目输出唯一。

`encode_e2m1_stochastic_sat()` 使用同一对边界，但以距离比例随机选择。`encode_e2m1()`
按配置作分派，语义与 E4M3 的外层分派一致。

### 为什么 NVFP4 需要 global + local 两层 scale

E2M1 最大值只有 6，若只靠每个 block 的 E4M3 scale，scale 自身在全张量范围很大时会被
压缩得不够精细。NVFP4 先用整张矩阵的 `amax` 建立 FP32 global decode scale：

$$
s_g=amax_{tensor}/(448\times6)=amax_{tensor}/2688.
$$

随后每个 rowwise 16 元素 block 再计算 E4M3 local scale：

$$
s_b=Q_{\mathrm{E4M3,RNE}}\left(amax_{block}/(6s_g)\right).
$$

元素实际使用的 combined scale 是 $s_b s_g$。于是 global scale 处理整张张量的粗范围，
local scale 让每个 16 元素 block 再贴近自身范围，E2M1 payload 最终编码
$z_i=x_i/(s_b s_g)$。

代码将 `global_scale` 保存为**解码方向**的 $s_g$，而非它的倒数。这使反量化直接是：

$$
\hat{x}_i=\operatorname{decode\_e2m1}(q_i)
\times\operatorname{decode\_e4m3}(s_b)\times s_g.
$$

`compute_nvfp4_global_scale()` 对全零张量返回 `1.0F`，避免后续除零；
`compute_nvfp4_local_scale_code()` 始终用 E4M3 RNE 编码 local scale，**不受元素
`rounding` 配置影响**。这是因为 `rounding` 的实验维度只应改变 E2M1 payload，不应让
scale 同时随机变化。

若 block 的 local scale 解码为零（全零 block 或 E4M3 scale 下溢），
`encode_nvfp4_element()` 会返回正零 nibble，避免除零并遵循项目的全零约定。

### 4 bit 真实打包

CPU 和 CUDA 都不以一个 `uint8_t` 存一个 E2M1 元素。两个逻辑 nibble 必须合到一个物理
byte：

```text
线性元素下标:  2k            2k + 1
逻辑 nibble:    low            high
物理 byte:      bits [3:0]     bits [7:4]
```

`pack_e2m1_nibbles(low, high)` 做掩码与左移，得到
`(low & 0x0f) | ((high & 0x0f) << 4)`；两个 `unpack_*` 函数是精确反操作。
元素总数为奇数时，最后一个 byte 的 high nibble 传零并保留为零。这个约定对跨行配对也
成立：配对依据是 row-major 的**全局线性下标**，不是“同一行的相邻列”。

## 接口到量化流程的对应关系

| 格式 | 上层先计算 | 每元素调用 | 每元素反量化 |
| --- | --- | --- | --- |
| MXFP8 block | 每个 32 元素 block 的 `compute_mxfp8_scale_code()` | `encode_mxfp8_element()` | `decode_mxfp8_element()` |
| MXFP8 tensor | 全矩阵一次 `compute_mxfp8_scale_code()` | `encode_mxfp8_element()` | `decode_mxfp8_element()` |
| NVFP4 | 一次 `compute_nvfp4_global_scale()`，每 16 元素 block 一次 `compute_nvfp4_local_scale_code()` | `encode_nvfp4_element()` 后两两 `pack_e2m1_nibbles()` | unpack 后 `decode_nvfp4_element()` |

这也解释了 CUDA kernel 的职责边界：kernel 负责并行 `amax` 规约、计算正确的 block
或 tensor scale 下标、读写 payload；codec 负责每个元素的数值语义。若要排查“payload
与 CPU reference 不一致”，先检查传入 codec 的 scale、线性下标随机数和元素值，再检查
codec 本身。

## 实现中的边界策略速查

| 情况 | E4M3 | E8M0 | E2M1 | 完整 pipeline |
| --- | --- | --- | --- | --- |
| `+0` / `-0` 写入 | 统一 `0x00` | scale 零写 `0x00`，解码为 $2^{-127}$ | 统一 `0x0` | 全零 block payload 都为正零。 |
| 外部文件中的 `-0` | decoder 保留 | 不适用 | decoder 保留 | 合法，可读回。 |
| 有限溢出 | 饱和到 `±448` | 向上取整后夹到最大有限 scale | 饱和到 `±6` | 正常量化输入先要求有限。 |
| NaN | canonical `0x7f` | `0xff` | 无 code，单元素 fallback 为 0 | 原始输入直接报错。 |
| Inf | 饱和到 `±448` | `0xff` | 饱和到 `±6` | 原始输入直接报错。 |

这里“单元素 codec 的 fallback”不等于端到端文件语义。codec 必须对任何 bit 输入有稳定
行为，便于 host/device 测试；真正的量化接口会在 scale 规约前拒绝非有限输入，避免把
这些 fallback 写入正常 QDWGT 产物。

## 建议的源码阅读顺序

1. 先读 `fp32_utils.cuh` 的 `float_to_bits()`、`floor_log2_positive()` 和
   `power_of_two()`，理解为什么 subnormal 需要单独处理。
2. 读 `decode_e4m3()`，用 E4M3 的 normal/subnormal 公式手算几个 code。
3. 读 `find_e4m3_lower_magnitude_code()`、`encode_e4m3_rne_sat()` 与
   `encode_e4m3_stochastic_sat()`，把“下界、上界、RNE/SR”串起来。
4. 读 E8M0 与 `compute_mxfp8_scale_code()`，再读 MXFP8 的两项组合接口。
5. 读 E2M1 codebook 与 `encode_e2m1_*()`；它更小，适合验证对 subnormal 与中点的理解。
6. 最后读 NVFP4 的 global/local scale 和 nibble pack/unpack，再进入
   `src/reference/` 与 `src/cuda/` 看矩阵级调度。
