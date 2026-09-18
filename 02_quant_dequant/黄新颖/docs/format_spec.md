# MXFP8 与 NVFP4 数值格式规范

本文是本项目量化、反量化、测试和二进制文件字段的**唯一数值语义来源**。实现必须遵循本文的位编码、scale 含义和舍入约定；如果以后需要修改这些规则，必须同时更新本文、测试向量和文件版本号。

若希望从代码角度理解每个 bit-field helper、RNE/SR 编码分支和 nibble 打包，请阅读
[format_codecs.md](format_codecs.md)。该文档是本规范的源码导读，不改变本文的数值约定。
CUDA kernel 如何并行计算这些 scale、编码 payload 与解包反量化，见
[kernels.md](kernels.md)。

## 范围与资料来源

- **MXFP8** 采用 [OCP Microscaling Formats (MX) Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) 的 MXFP8 语义。本项目基线固定为 **E4M3 元素 + E8M0 scale + 32 元素 block**。OCP 也定义 E5M2 变体，但题目的配置文件没有元素格式字段，因此 E5M2 不纳入第一版。
- **NVFP4** 采用 NVIDIA [Transformer Engine NVFP4 文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html) 的一维双层缩放语义：**E2M1 元素 + 每 16 元素一个 E4M3 局部 scale + 整张张量一个 FP32 全局 scale**。它不是 OCP 的 MXFP4：后者使用 32 元素 block 和 E8M0 scale。
- MXFP8 的 scale 计算采用 NVIDIA [TensorRT MX-compliant dynamic quantization](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html#mx-compliant-dynamic-quantization) 的 `round_up_to_e8m0(amax / 448)`。这是 OCP 建议算法允许的实现选择，能保证 block 最大值不会因 scale 偏小而溢出。
- 本项目在 RTX 4060（SM89）上使用普通 CUDA kernel 软件模拟格式；不依赖 Blackwell 的原生 MXFP8/NVFP4 指令。

## 共同约定

### 输入、布局和 block

- 输入 FP16 或 FP32 时，先转换为 FP32 进行 scale 计算和编码；反量化再转换到配置指定的 FP16、BF16 或 FP32。
- 基线布局是 **rowwise**：对形状为 `R × C` 的 row-major 矩阵，每一行独立分为连续的 `1 × block_size` block。MXFP8 的 `block_size = 32`，NVFP4 的 `block_size = 16`。
- 一行末尾不足一个 block 时，逻辑上以零补齐以便定义 block，但 payload 只保存原始 `R × C` 个元素。补零不参与有效元素输出，也不改变 `amax`。
- 因此 block 模式下 scale 数量是 `R * ceil(C / block_size)`，而不是跨行展平后的 `ceil(R * C / block_size)`。
- 本项目第一版只接受**有限值**输入。输入含 NaN 或 `±Inf` 时，量化接口必须返回错误，不写任何输出文件。这样可避免 NVFP4 无法表示 NaN/Inf 时引入未定义的私有 sentinel。

### 通用舍入

配置字段 `rounding` 只控制**元素**量化；scale 的量化规则在各格式章节单独规定。

- `nearest`：round-to-nearest, ties-to-even（RNE，最近值且中点取偶）。这是 OCP MX 对 FP8/FP4 转换要求支持的基线舍入。
- `stochastic`：设非负幅值 `a` 位于相邻可表示幅值 `l <= a <= h` 之间。以概率 `(a - l) / (h - l)` 选择 `h`，以剩余概率选择 `l`；精确命中、`l == h` 和饱和边界的结果不依赖随机数。最后恢复原符号。
- 超过最大有限值时，两种模式都饱和到最大有限值；小于最小 subnormal 的幅值在 `nearest` 下为零，在 `stochastic` 下可按零与最小 subnormal 的距离随机选择。

### stochastic 的确定性随机数映射

`stochastic_seed` 不是任意 PRNG 实现的提示，而是本项目数值语义的一部分。对 row-major 全局线性下标为 `n` 的元素，CPU reference 与未来 CUDA kernel 必须使用下列无状态映射生成 $u_n\in[0,1)$。无状态表示每个元素只依赖 seed 和自身下标；前一个元素是否恰好命中可表示值，不会改变后一个元素的随机数。

所有 `uint64_t` 加法和乘法按模 $2^{64}$ 回绕。令 $\gamma=\texttt{0x9e3779b97f4a7c15}$，先计算 `state = stochastic_seed + gamma * (n + 1)`，再执行：

```cpp
std::uint64_t splitmix64(std::uint64_t state) {
    state += 0x9e3779b97f4a7c15ULL;
    state = (state ^ (state >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    state = (state ^ (state >> 27U)) * 0x94d049bb133111ebULL;
    return state ^ (state >> 31U);
}

const std::uint64_t random_bits = splitmix64(state);
const std::uint32_t random_mantissa =
    static_cast<std::uint32_t>(random_bits >> 40U);
const float uniform_random =
    static_cast<float>(random_mantissa) * 0x1.0p-24F;
```

`random_mantissa` 取 `random_bits` 的高 24 bit，范围为 $[0,2^{24}-1]$，因此 `uniform_random` 严格位于 $[0,1-2^{-24}]$，不会等于 `1.0F`。`nearest` 模式不使用该映射；`stochastic` 模式将该值传给格式特有的元素编码器。即使实现提前计算了某个 $u_n$，由于没有可变 PRNG 状态，也不影响其他元素。

### 为什么低精度浮点仍然需要 scale

FP8/FP4 的元素不是 INT 量化里的整数 code，但它们仍只有很小的有限数值范围。例如
E4M3 的有限幅值范围是从 $2^{-9}$ 到 $448$，E2M1 的有限幅值范围则只有从 $0.5$ 到
$6$。如果直接把原始 FP32 输入送进这些格式：

- 当输入幅值大于格式最大值时，编码只能饱和，较大的数被压成同一个最大值；
- 当输入幅值远小于格式的最小 normal/subnormal 时，编码会进入 subnormal，甚至变为零；
- 一个张量的不同区域通常有不同量级。让一个很大的异常值决定所有小值的尺度，会使小值
  被推到低精度格式最稀疏的区域。

因此量化不是直接计算 $p_i=Q(x_i)$，而是先为一组元素选正 scale：

$$
z_i=x_i/s,\qquad p_i=Q(z_i),\qquad \hat{x}_i=\operatorname{decode}(p_i)s.
$$

scale 的数学任务是把这组数据的有效范围平移到低精度 codebook 的有效范围。若元素格式
最大有限幅值为 $q_{\max}$，而当前组的 $amax$ 是
$a=\max_i|x_i|$，不饱和的必要条件是：

$$
\frac{a}{s}\le q_{\max}
\qquad\Longleftrightarrow\qquad
s\ge\frac{a}{q_{\max}}.
$$

在满足覆盖范围的 scale 中，通常希望选尽可能小的那个：scale 越小，归一化后的
$z_i$ 越大，小元素越不容易落入 subnormal/零附近；scale 过大则会浪费低精度格式顶部的
许多可表示值。block scale 的意义正是在每个小块各自选接近该下界的 scale，而非让整张
张量的全局最大值主导所有块。

以 E4M3 为例，若整个张量有 $amax=256$，但某个 32 元素 block 的 $amax=1$：

- tensor scale 的最小 E8M0 选择是 $s_t=1$；该 block 的 $x=0.01$ 会变为
  $z_t=0.01$，已落在 E4M3 的 subnormal 区；
- block scale 的 E8M0 选择是 $s_b=2^{-8}$；同一个元素变为 $z_b=2.56$，落在
  normal 区，拥有更好的相对分辨率。

代价是 block mode 需要额外存储 scale，并在量化时做更多次 $amax$ 规约；这就是
压缩率、kernel 工作量与数值精度之间的基本取舍。

## MXFP8-E4M3

### 表示形式

一个 MXFP8 block 由 1 个 E8M0 scale 字节和 32 个 E4M3 元素字节组成。反量化语义为：

$$
\hat{x}_i = p_i \times s_b
$$

其中 $p_i$ 是 E4M3 私有元素，$s_b$ 是该 block 的 E8M0 scale。

| 字段 | 位宽 | 规则 |
| --- | ---: | --- |
| E4M3 符号 `S` | 1 | bit 7。 |
| E4M3 指数 `E` | 4 | bits `[6:3]`，bias 为 7。 |
| E4M3 尾数 `M` | 3 | bits `[2:0]`。 |
| E8M0 指数 `e` | 8 | 一个无符号字节，bias 为 127。 |

对 E4M3 的普通编码：当 `E > 0` 时，

$$
p=(-1)^S\,2^{E-7}\,(1+M/8)
$$

当 `E = 0` 时为 subnormal：

$$
p=(-1)^S\,2^{-6}\,(M/8)
$$

E4M3 关键值：最小 subnormal 为 $2^{-9}$，最小 normal 为 $2^{-6}$，最大有限值为 `448`（位模式 `0x7e`），`E=15, M=7` 为 NaN。E4M3 没有 Inf 编码；对有限输入溢出使用饱和规则。

项目写入策略会把任意有限 `+0` 或 `-0` 元素编码为 canonical 正零 `0x00`，无论该元素所在 block 是否全零；这使量化器的输出唯一。解码器仍接受外部 QDWGT 中合法的 `0x80`，并将其还原为 FP32 `-0`。因此“写入时规范化”和“读取时保留 bit pattern”是两个不同的规则。

E8M0 的有限 scale 为：

$$
s_b = 2^{e-127},\quad e\in[0,254]
$$

`e=255` 是 NaN，E8M0 没有零或 Inf。解码损坏文件时，若 E8M0 为 NaN，则按 OCP MX 语义将该 block 的所有输出视为 NaN；量化器不会为有限输入产生该编码。

### block scale 计算

对第 `b` 个有效 block，先计算：

$$
a_b=\max_i |x_i|, \qquad r_b = a_b / 448
$$

然后按向上取整到 E8M0 的幂次：

$$
e_b=\operatorname{clamp}(\lceil\log_2(r_b)\rceil+127,0,254),
$$

$$
s_b=2^{e_b-127}.
$$

#### 为什么 MXFP8 选择 E8M0 向上取整的 block scale

E4M3 的最大有限幅值是 $448$。对当前 block，若其实际 decode scale 是 $s_b$，最大
归一化幅值为 $a_b/s_b$。为了保证最大元素不因 E4M3 上溢而饱和，必须满足：

$$
\frac{a_b}{s_b}\le448
\qquad\Longleftrightarrow\qquad
s_b\ge\frac{a_b}{448}.
$$

理想的连续 scale 是 $a_b/448$，但 E8M0 没有 mantissa，只能表示 $2$ 的整数次幂。
所以项目采用不小于理想值的最小 E8M0 幂：

$$
s_b=2^{\lceil\log_2(a_b/448)\rceil}.
$$

这就是向上取整到 E8M0，并在实际编码时加入 E8M0 可表示指数范围的 clamp。若没有触及
clamp 且 $a_b>0$，它还给出一个很有用的范围：

$$
224 < \frac{a_b}{s_b}\le448.
$$

也就是说，block 中最大元素一定落在 E4M3 顶部一半的有限动态范围，而不会溢出；这比
向下取整安全得多。若错误地选择 $s_b<a_b/448$，则 $a_b/s_b>448$，最大元素必然饱和，
并破坏该 block 的数值关系。

E8M0 只能取 2 的幂也带来一个性质：对仍位于 E4M3 normal 区的元素，乘除 $s_b$ 在
二进制语义上主要是指数移动，不再额外引入一个任意 mantissa 的 scale 近似误差。代价是
scale 最多可能比理想连续值大接近 2 倍，因此 block 内远小于 $a_b$ 的元素仍可能进入
subnormal；这正是把 block 设为 32 而不是整张张量的原因。

$a_b = 0$ 时写入 $e_b = 0$，即最小有限 E8M0 scale $2^{-127}$，并把所有元素编码为正零。由于所有私有元素均为零，反量化结果仍为零。

当 $a_b>0$ 但理论指数小于 E8M0 的最小指数时，上式中的 clamp 同样令 $e_b=0$。这仍表示 $2^{-127}$，而不是零：E8M0 没有零编码。有限输入过小导致 `a_b / 448` 在 FP32 中下溢为零时，当前实现也按这一最小有限 scale 处理。

等价实现可以直接检查 FP32 的指数和 mantissa：对 $r_b$ 提取指数；若 mantissa 非零则指数加一。这正是 NVIDIA 对 E8M0 向上取整的描述。

### 编码、解码与 scale_mode

**严格 MXFP8 block 模式**：

1. 按上节得到每个 `1 × 32` block 的 E8M0 $s_b$。
2. 计算 $z_i = x_i / s_b$。
3. 按配置的元素舍入，将 $z_i$ 转为 E4M3；有限溢出饱和到 `±448`。
4. 反量化为 $\operatorname{decode\_e4m3}(p_i) \times s_b$。

`scale_mode = "tensor"` 是为了满足题目配置加入的**项目扩展**，不再是严格 OCP MXFP8：在整张 `R × C` 矩阵上计算一个 `amax`，按同一公式只生成一个 E8M0 scale，所有元素共享它。文件 header 必须保存 `scale_mode`，读取方不得把 tensor 模式数据当作 OCP block 模式数据。

### 可用于单元测试的向量

一个 block 含有 `{0, 1, -1, 448}`，其余元素为零时：

- `amax = 448`，所以 $s_b = 1$，E8M0 字节为 `0x7f`。
- 对应 E4M3 字节依次为 `0x00`、`0x38`、`0xb8`、`0x7e`。

当前 CPU reference 的端到端测试还覆盖一个 `1 × 33` 矩阵：前 32 个元素中前四项为 `{0, 1, -1, 448}`、其余为零，尾 block 的唯一元素为 `-896`。两个 local scale 依次为 `0x7f`（$1$）和 `0x80`（$2$）；第 33 个 E4M3 payload 为 `0xfe`（$-448$）。该向量同时验证 rowwise 尾 block、scale 索引和 QDWGT 文件 round-trip。

## NVFP4-E2M1

### 表示形式

NVFP4 的一个元素由 E2M1 nibble、局部 E4M3 scale 和全局 FP32 scale 共同解释：

$$
\hat{x}_i = q_i \times s_b \times s_g
$$

其中 $q_i$ 是 E2M1，$s_b$ 是该 `1 × 16` block 的 E4M3 局部 decode scale，$s_g$ 是整张张量的 FP32 全局 decode scale。这个公式与 NVIDIA Transformer Engine 文档一致。

E2M1 的 bit layout 为 `S EE M`：符号位 `S` 为 bit 3，指数 `EE` 为 bits `[2:1]`，尾数 `M` 为 bit 0，指数 bias 为 1。其正数幅值 codebook 如下：

| `EE M` | 幅值 | 说明 |
| --- | ---: | --- |
| `000` | 0 | 零；符号位可形成 `+0`、`-0`。 |
| `001` | 0.5 | subnormal，也是最小正数。 |
| `010` | 1 | normal。 |
| `011` | 1.5 | normal。 |
| `100` | 2 | normal。 |
| `101` | 3 | normal。 |
| `110` | 4 | normal。 |
| `111` | 6 | 最大有限值。 |

因此 E2M1 不存在 Inf 或 NaN 编码，最大幅值 $q_{\max} = 6$。局部 scale 使用与上文相同的 OCP E4M3 编码：正的有限 E4M3 scale 范围是 `0` 至 `448`，并以一个字节存储。

### 双层 scale 计算

本项目存储和使用的是**解码方向**的 $s_g$，不存倒数：

$$
a_g=\max_i |x_i|,
\qquad
s_g=
\begin{cases}
a_g/(448\times6), & a_g>0 \\
1, & a_g=0
\end{cases}
$$

对每一个 16 元素 block：

$$
a_b=\max_i|x_i|,
\qquad
r_b=a_b/(6s_g),
\qquad
s_b=\operatorname{E4M3\_RNE\_SAT}(r_b).
$$

`E4M3_RNE_SAT` 表示 E4M3 scale 始终用 RNE 转换、超过 `448` 时饱和；它不受元素 `rounding` 配置影响。量化元素时：

$$
z_i=x_i/(s_b s_g),
\qquad
q_i=\operatorname{E2M1}_{\text{rounding}}(z_i).
$$

#### 为什么 NVFP4 还需要 FP32 global scale

NVFP4 的元素 E2M1 最大幅值只有 $6$；一个 local scale 虽可用 E4M3 表示，最大有限
值也只有 $448$。因此，对于给定 FP32 global decode scale $s_g$，某个 block 可覆盖的
最大原始幅值至多是：

$$
q_{\max}\times s_{b,\max}\times s_g
=6\times448\times s_g
=2688s_g.
$$

令整张张量的最大幅值为 $a_g$。要让任意 block 的理想 local-scale 目标不超出 E4M3
的最大值，必须至少有 $2688s_g\ge a_g$。项目正好选择最小满足该覆盖条件的 FP32
global scale：

$$
s_g=\frac{a_g}{2688}.
$$

这一步的作用不是把所有元素直接压进 E2M1；如果只使用这个 global scale，E2M1 单独的
可覆盖范围仍仅为 $6s_g=a_g/448$，绝大多数大元素依然会饱和。它先把整张张量的粗粒度
动态范围映射到“E4M3 local scale 最大可达 448”的坐标系。

对第 $b$ 个 block 的 $a_b$，代入上述 $s_g$ 后，理想 local scale 为：

$$
s_b^\ast=\frac{a_b}{6s_g}
=448\frac{a_b}{a_g}.
$$

因为 $0\le a_b\le a_g$，必有 $0\le s_b^\ast\le448$。这正是 global scale 的关键数学
效果：无论各 block 的绝对量级多大，local scale 的理想值都会落入 E4M3 的有限范围。
随后 E4M3 用 RNE 将 $s_b^\ast$ 编码为实际 local scale $s_b$，再令
$z_i=x_i/(s_b s_g)$。若没有 local-scale 的 E4M3 舍入误差，block 最大元素会满足
$a_b/(s_b^\ast s_g)=6$，恰好用满 E2M1 的正最大值。

E4M3 local scale 是离散的，RNE 后的 $s_b$ 可能略大或略小于 $s_b^\ast$；因此它优化的是
局部范围匹配，而不是像 MXFP8 E8M0 向上取整那样严格保证每个元素绝不饱和。实际编码器
仍对 E2M1 的超范围值执行饱和。这是 NVFP4 用更小 4-bit payload 换取更高压缩率时的
固有误差来源之一。

如果没有 global scale，而直接要求 E4M3 local scale 吸收原始范围，则任一
$a_b>6\times448=2688$ 的 block 都需要超过 E4M3 最大值的 local scale，必然发生 scale
饱和。即使没有这么大的 block，global + local 的分工仍然有价值：$s_g$ 表示整张张量的
粗粒度幅值，$s_b$ 则把每个 16 元素 block 拉回 E2M1 最有效的 $[-6,6]$ 附近。

若 $a_b = 0$，局部 E4M3 scale 写为正零，16 个 nibble 都写为零。若一个非零 block 的 $s_b$ 因 E4M3 下溢而成为零，则该 block 的所有元素量化为零；这是有限 E4M3 scale 分辨率导致的预期结果。

计算时可以使用编码方向的倒数 $g_{\mathrm{encode}} = 1 / s_g = 2688 / a_g$ 避免除法，但文件中的 `global_scale` 字段必须始终保存上述解码方向的 $s_g$。这一命名约定避免与部分 NVIDIA API 中使用的“编码 scale”混淆。

NVFP4 在本项目中严格采用 block scaling：一个 FP32 $s_g$ 加上每个 `1 × 16` block 的一个 E4M3 $s_b$。因此 `format = "nvfp4"` 时 `scale_mode` 必须为 `"block"`；`"tensor"` 会在配置读取、`QuantizedTensor` 元数据校验和 QDWGT 文件读取时被拒绝。这既符合 NVIDIA 的标准 NVFP4 recipe，也避免将分层缩放错误地退化成单个 local scale。

### 4-bit 打包

题目要求真实 4-bit 存储；本项目的 payload 约定为线性 row-major 下标 `n = row * C + col`：

$$
payload[n/2] =
\begin{cases}
(payload[n/2]\ \&\ 0xf0)\ |\ q_n, & n\text{ 为偶数} \\
(payload[n/2]\ \&\ 0x0f)\ |\ (q_n \ll 4), & n\text{ 为奇数}.
\end{cases}
$$

- 偶数元素在低 nibble，奇数元素在高 nibble。
- payload 字节数为 `ceil(R * C / 2)`；如果元素总数为奇数，最后一个字节的高 nibble 必须清零。
- CUDA kernel 必须以字节为单位执行 packed load/store，并通过掩码和移位读取两个 E2M1 元素，不能用一个 `uint8_t` 保存一个 nibble。
- NVIDIA 的硬件 GEMM 还要求 scale swizzle 和部分对齐填充；本项目是软件模拟且只做量化/反量化，因此 payload 与 scale 按上述朴素 row-major 顺序保存，不实施硬件 swizzle。

### 可用于单元测试的向量

假设整张张量与当前 block 的 `amax = 6`：

- $s_g = 6 / 2688 = 1 / 448$。
- $s_b = \operatorname{E4M3}(448)$，字节为 `0x7e`。
- `+6` 的 E2M1 nibble 为 `0x7`，`-6` 为 `0xf`。
- row-major 连续元素 `{+6, -6}` 的 packed 字节为 `0xf7`，反量化分别为 `+6`、`-6`。

## 配置、文件和兼容性要求

| 配置 | MXFP8 | NVFP4 |
| --- | --- | --- |
| `format` | `mxfp8`，基线为 E4M3 + E8M0。 | `nvfp4`，E2M1 + E4M3 + FP32 global scale。 |
| `block_size` | 配置和 header 均必须为 32；block 模式按此大小分组。 | 配置和 header 均必须为 16；按此大小分组。 |
| `scale_mode=block` | 严格 OCP MXFP8。 | NVIDIA 一维 NVFP4 recipe。 |
| `scale_mode=tensor` | 单 E8M0 scale 的项目扩展。 | **不支持**；配置和 QDWGT header 均会被拒绝。 |
| `rounding` | 控制 E4M3 元素转换。 | 控制 E2M1 元素转换。 |

低精度权重文件的 header 至少要写入：`format`（QDWGT v1 的一个字节：`1 = MXFP8`、`2 = NVFP4`）、`scale_mode`、矩阵形状、block size、rounding、payload 字节数、局部 scale 个数与类型，以及 NVFP4 的 FP32 `global_scale`。`mxfp8_e4m3`、`nvfp4_e2m1` 只是本文对数值 profile 的描述，不是 header 中保存的字符串。物理 header 字节布局、枚举值与 section 顺序以 [file_format.md](file_format.md) 为准。

## 实现和测试边界

- MXFP8 CPU reference 的**量化与反量化方向**已经实现：量化支持 `block`、`tensor`、`nearest`、`stochastic`，会构造 `QuantizedTensor` 并可写入 QDWGT；反量化按相同的 tensor 或 rowwise block scale 索引解码为 host FP32，并由 `DequantizationConfig` 指定后续 QDTENSOR 的 FP16、BF16 或 FP32 物理输出类型。
- MXFP8 CUDA 的 block/tensor 量化和反量化已实现，并在有 CUDA device 时与 CPU reference 对照 payload、E8M0 scale 和 FP32 反量化结果；测试覆盖 tail block、grid-stride 路径、nearest、stochastic 和非有限输入拒绝。QDWGT I/O 还覆盖了 `NVFP4 + tensor` header 的拒绝。
- NVFP4 的 CPU reference 与 CUDA 量化/反量化已经实现：量化先用两阶段规约得到 FP32 global scale，再用 16-lane tile 计算每个 E4M3 local scale，最后用连续 32 元素 warp 让偶数 lane 独占写 packed E2M1 byte。反量化不需要规约，一个线程读取一个 physical payload byte，解包 low/high nibble 后分别用自己的 `(row, column)` 推导 rowwise local-scale 下标，因此能正确处理奇数列的跨行 byte 与奇数元素的尾 nibble。CPU 测试覆盖 E2M1 codebook、双层 scale、跨行 byte 配对、stochastic 可复现性、反量化与非有限输入拒绝；有 CUDA device 时，测试逐元素对照 CPU/CUDA 量化与反量化结果，覆盖三种输出 dtype、persistent grid-stride、尾 block 和 `NVFP4 + tensor` 非法组合。
- CPU reference 和 CUDA kernel 必须调用或复制同一套数学规则；对 `nearest`，两者需要做到 payload 与 scale 位级一致；对 `stochastic`，在相同随机种子与随机数映射下也需要一致。
- RTX 4060 仅用于软件模拟和性能评测。它不支持 Blackwell 的原生 MXFP8/NVFP4 Tensor Core 路径，因此报告不得把本项目性能称作硬件格式指令性能。
