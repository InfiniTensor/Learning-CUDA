## 测试
~/Learning-CUDA目录下

```bash
# 默认 bf16
make nf4

# 显式指定 dtype
make nf4 DTYPE=fp16
make nf4 DTYPE=bf16

# 简写（等价于上面的 DTYPE）
make nf4 fp16
make nf4 bf16

# 指定 GPU
make nf4 CUDA_DEVICE=7
make nf4 fp16 CUDA_DEVICE=7

# 查看用法
make help
```

说明：
- `make nf4 fp16` 中的 `fp16` 是 Make 目标简写，Makefile 已兼容并自动映射为 `DTYPE=fp16`。
- 运行阶段会传参给 `mainla`，输出路径随 dtype 自动切换到 `data/output_fp16.bin` 或默认 bf16 输出文件。
## NF4反量化
权重 = NF4表值 × (一级scale × 二级scale) + offset
scale = code2[ absmax_q[block] ] × absmax2[group]
w =
  NF4_TABLE[q]
× code2[ absmax_q[block] ]
× absmax2[group]
+ offset
关联关系：
Group (16384 weights)
 └── 256 Blocks
      └── 64 weights
           └── 2 per byte
## bitsandbytes参考实现
```shell
source /data/shared/miniconda3/etc/profile.d/conda.sh && conda activate cuda && python - <<'PY'
import torch
import bitsandbytes as bnb

rows, cols, blocksize = 16384, 16384, 64

# Generate data
x = torch.randn(rows, cols, device='cuda', dtype=torch.bfloat16)
packed, qstate = bnb.functional.quantize_4bit(
    x,
    blocksize=blocksize,
    quant_type='nf4',
    compress_statistics=True
)

# Warmup
for _ in range(5):
    y = bnb.functional.dequantize_4bit(packed, qstate, quant_type='nf4', blocksize=blocksize)

torch.cuda.synchronize()

# Timing
iters = 100
start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(iters):
    y = bnb.functional.dequantize_4bit(packed, qstate, quant_type='nf4', blocksize=blocksize)
stop.record()
stop.synchronize()

ms = start.elapsed_time(stop) / iters

# Approx bandwidth
num_elements = rows * cols
size_packed = num_elements // 2
num_blocks = (num_elements + blocksize - 1) // blocksize
num_groups = (num_blocks + 255) // 256
size_absmax_q = num_blocks
size_absmax2 = num_groups * 2
size_code2 = 256 * 2
size_out = num_elements * 2

total_bytes = size_packed + size_absmax_q + size_absmax2 + size_code2 + size_out
bandwidth = total_bytes / (ms / 1000.0) / 1e9

print(f"bitsandbytes dequantize_4bit: {ms:.6f} ms")
print(f"Approx bandwidth: {bandwidth:.2f} GB/s")
PY
```
bitsandbytes dequantize_4bit: 1.241162 ms
Approx bandwidth: 544.10 GB/s
## 构造标准测试集





## 编写main.cu
### 反量化查表
来自github bitsandbytes/csrc/kernels.cu
```cpp
__device__ static float fp4_dequantization_lut[8] = {
    0.0f,            // 0b000
    0.005208333333f, // 0b001
    0.66666667f,     // 0b010
    1.0f,            // 0b011
    0.33333333f,     // 0b100
    0.5f,            // 0b101
    0.16666667f,     // 0b110
    0.25f            // 0b111
};

__device__ static float nf4_dequantization_lut[16] = {
    -1.0f,                 // 0b0000
    -0.6961928009986877f,  // 0b0001
    -0.5250730514526367f,  // 0b0010
    -0.39491748809814453f, // 0b0011
    -0.28444138169288635f, // 0b0100
    -0.18477343022823334f, // 0b0101
    -0.09105003625154495f, // 0b0110
    0.0f,                  // 0b0111
    0.07958029955625534f,  // 0b1000
    0.16093020141124725f,  // 0b1001
    0.24611230194568634f,  // 0b1010
    0.33791524171829224f,  // 0b1011
    0.44070982933044434f,  // 0b1100
    0.5626170039176941f,   // 0b1101
    0.7229568362236023f,   // 0b1110
    1.0f                   // 0b1111
};
```
bitsandbytes使用device，考虑是否使用constant
| 维度      | **device** static | **constant**      |
| ------- | ----------------- | ----------------- |
| 存储位置    | Global Memory     | Constant Memory   |
| 缓存      | L2 / L1           | 专用 Constant Cache |
| Warp 广播 | 无              |  有               |
| 访问延迟    | 高                 | 低                 |
| 适合      | 普通全局数据            | 查表 / 常量           |
使用constant先。
### host逻辑
1.输入解析，读取二进制文件
2.内存规划
3.数据加载，分配显存
4.启动kernel
5.记录性能，写入数据
### device逻辑
1. 全局一维线程索引
2. 读取这 1 个字节，并解包成两个 4-bit 索引
3. 计算当前字节属于哪一个量化 Block 和 Group
4. 暴力从全局内存 (Global Memory) 读取双重量化参数
5. 结合 NF4 查表，计算真实的浮点权重
6. 最朴素的分别写回 (没有使用 Union 向量化合并写入)

## nativekernel问题解决
精度远超阈值
### 解码索引覆盖不足
最初 grid 维度按 num_groups 计算，实际应按 packed byte 数计算，导致大部分元素没写。
```
Kernel Time: 0.121476 ms
Effective Bandwidth (approx): 347.457 GB/s
Output written to nf4/data/output.bin
```

### gendata，offset 处理错误
offset 并非恒为 0，而且应加在 absmax 上，不应加在最终权重上。
```
Kernel Time: 0.141412 ms
Effective Bandwidth (approx): 298.473 GB/s
Output written to nf4/data/output.bin
MAE (Mean Absolute Error): 0.000024
Max Error:                 0.031250
------------------------------
✅ PASS: MAE (0.000024) is within threshold (0.01)
```
### 为了可拓展性，使用 Grid-Stride Loops
```cpp
int blockSize = 256;
// 我们根据硬件 SM 数量来决定 grid，或者简单给一个足够大的数
int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
int gridSize = 32 * numSMs; // 保证每个 SM 都有活干
```
修改代码
```cpp
int sm_count = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    int grid_x = sm_count * 4;
    int64_t max_grid = (total_bytes + blockDim.x - 1) / blockDim.x;
    if (grid_x > max_grid) {
        grid_x = static_cast<int>(max_grid);
    }
    if (grid_x < 1) {
        grid_x = 1;
    }
    dim3 gridDim(grid_x);
```
此时性能
```
Kernel Time: 0.121476 ms
Effective Bandwidth (approx): 347.457 GB/s
```
换成通用api计算
```cpp
int sm_count = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    int grid_x = sm_count * 4;
    int64_t max_grid = (total_bytes + blockDim.x - 1) / blockDim.x;
    if (grid_x > max_grid) {
        grid_x = static_cast<int>(max_grid);
    }
    if (grid_x < 1) {
        grid_x = 1;
    }
    dim3 gridDim(grid_x);
```
此时性能
```
SM count: 108, max active blocks/SM: 8, grid_x: 864
Kernel Time: 0.124518 ms
Effective Bandwidth (approx): 338.969 GB/s
Output written to nf4/data/output.bin
```
## 优化大纲
1. 向量化访存优化
```cpp
output[tid * 2]     = __float2bfloat16(w1_fp32);
    output[tid * 2 + 1] = __float2bfloat16(w2_fp32);
```
这发起了两次独立的写入，gpu显存控制器可以一次性吃32-bit，64-bit，甚至128-bit的数据

2. 冗余计算与重复访存
blocksize = 64，那么 32 个线程（正好是一个 Warp）处理的 64 个权重，其实都属于同一个 Block。
3. 整数除法（位运算优化）
4. code2 码表的读取延迟
## 实际优化
### 向量化访存
向量化加载 packed_weights
```cpp
uint8_t packed = packed_weights[byte_idx];
```
单字节 load，效率差。GPU 更适合 4B / 8B 对齐访问。
需注意，主机端对 packed buffer 做 4 字节对齐补零，避免最后一个 uint32 读取越界。
```cpp
for (int b = 0; b < 4; ++b) {
            int64_t byte_idx = word_idx * 4 + b;
            if (byte_idx >= total_bytes) {
                continue;
            }

            uint8_t packed = static_cast<uint8_t>((packed_word >> (8 * b)) & 0xFF);
            uint8_t idx1 = (packed >> 4) & 0x0F; // 高 4 位对应 output[byte_idx * 2]
            uint8_t idx2 = packed & 0x0F;        // 低 4 位对应 output[byte_idx * 2 + 1]

            int bytes_per_block = blocksize / 2;
            int block_id = static_cast<int>(byte_idx / bytes_per_block);
            int group_id = block_id / group_size;

            float a2 = __half2float(absmax2[group_id]);
            uint8_t qa = absmax_q[block_id];
            float c2 = __half2float(code2[qa]);
            float real_absmax = c2 * a2 + offset;

            float w1_fp32 = NF4_LUT[idx1] * real_absmax;
            float w2_fp32 = NF4_LUT[idx2] * real_absmax;

            int64_t out_idx = byte_idx * 2;
            if (out_idx < num_elements) {
                output[out_idx] = __float2bfloat16(w1_fp32);
            }
            if (out_idx + 1 < num_elements) {
                output[out_idx + 1] = __float2bfloat16(w2_fp32);
            }
        }
// Kernel Time: 0.172722 ms
// Effective Bandwidth (approx): 244.368 GB/s
```
#### 读取方式改为 16B 向量化

kernel 参数从 const uint32_t* 改为 const uint4*
每线程处理 1 个 16B word（32 个权重），total_words = ceil(total_bytes / 16)
读取 16B 后用 #pragma unroll 在寄存器里拆 16 个 byte，再拆成 32 个 4-bit 索引
参数只读一次

对应这 32 个权重只计算一次 block_id/group_id
只读一次 absmax2/absmax_q/code2/offset，得到 real_absmax，后面 32 个权重复用
写回向量化

每个byte 生成 2 个 bf16，打包成 1 个 uint32
16 个 uint32 组成 4 个 float4，一次写回 8 个 bf16
尾部不足 32 权重时走标量写回，避免越界
主机侧对齐与网格调整

packed buffer padding 改为 16B 对齐
cudaMalloc/cudaMemcpy 使用对齐后的大小
grid 上限按 total_words 而不是 total_bytes 计算
#### 此外，精简写回：跳过 float4 强转
```cpp
out_f4[0] = *reinterpret_cast<float4*>(&v0);
```
既然已经拼接好了 uint4，直接用 uint4 类型的指针写回即可。GPU 并不在乎你存入的是 float 还是 uint，只要它是 128 位的。当告诉 GPU 要执行一个存储（Store）操作时，硬件只需要知道两件事：
起始地址（Starting Address）： 数据要写到哪？
位宽（Bit Width）： 这一趟搬多大的数据（32位、64位还是 128位）？
当定义一个 uint4 变量并写回时，CUDA 编译器（NVCC）会生成一条类似 STG.E.128 (Store Global 128-bit) 的汇编指令。这条指令不管这 128 位里装的是 8 个 bfloat16、4 个 float，还是 16 个 char。它只负责把寄存器里的 128 个比特流，原封不动地拍到对应的显存地址上。
### 在native上实现合并写回（无向量访存）
```cpp
if (out_idx + 1 < num_elements) {
            Bf16Bits lo; lo.bf = __float2bfloat16(w1_fp32);
            Bf16Bits hi; hi.bf = __float2bfloat16(w2_fp32);
            out_u32[out_idx / 2] = static_cast<uint32_t>(lo.u) | (static_cast<uint32_t>(hi.u) << 16);
        } else if (out_idx < num_elements) {
            output[out_idx] = __float2bfloat16(w1_fp32);
        }
Kernel Time: 0.102705 ms
Effective Bandwidth (approx): 410.962 GB/s
```
## 加载表constant为广播可能conflict
```cpp
__shared__ float s_LUT[16];
    if (threadIdx.x < 16) {
        s_LUT[threadIdx.x] = NF4_LUT[threadIdx.x];
    }
    __syncthreads();
```
搬到 SM（流处理器）内部极其昂贵、速度极快的 Shared Memory（共享内存）中。
### 更改Shape: 16384x16384, Blocksize: 64
SM count: 108, max active blocks/SM: 8, grid_x: 864
Kernel Time: 0.968552 ms
Effective Bandwidth (approx): 697.243 GB/s
------------------------------
MAE (Mean Absolute Error): 0.000017
Max Error:                 0.031250
------------------------------
✅ PASS: MAE (0.000017) is within threshold (0.01)
## 使用inline(编译器优化)
```cpp
__device__ __forceinline__ float nf4_lut_value(uint8_t idx) {
    switch (idx & 0x0F) {
        case 0x0: return -1.0f;
        case 0x1: return -0.6961928009986877f;
        case 0x2: return -0.5250730514526367f;
        case 0x3: return -0.39491748809814453f;
        case 0x4: return -0.28444138169288635f;
        case 0x5: return -0.18477343022823334f;
        case 0x6: return -0.09105003625154495f;
        case 0x7: return 0.0f;
        case 0x8: return 0.07958029955625534f;
        case 0x9: return 0.16093020141124725f;
        case 0xA: return 0.24611230194568634f;
        case 0xB: return 0.33791524171829224f;
        case 0xC: return 0.44070982933044434f;
        case 0xD: return 0.5626170039176941f;
        case 0xE: return 0.7229568362236023f;
        default: return 1.0f;
    }
}
float v1 = nf4_lut_value(p >> 4) * real_absmax;
                float v2 = nf4_lut_value(p & 0x0F) * real_absmax;
Kernel Time: 2.58223 ms
Effective Bandwidth (approx): 261.524 GB/s
```
遂撤回，改为寄存器缓存表值


## 输出适配fp16
```cpp
template <typename T>
__device__ __forceinline__ T float_to_out(float v);

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_out<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ half float_to_out<half>(float v) {
    return __float2half(v);
}

template <typename T>
__device__ __forceinline__ uint32_t pack_pair_to_u32(float v1, float v2);

template <>
__device__ __forceinline__ uint32_t pack_pair_to_u32<__nv_bfloat16>(float v1, float v2) {
    __nv_bfloat162 packed = __floats2bfloat162_rn(v1, v2);
    return *reinterpret_cast<uint32_t*>(&packed);
}

template <>
__device__ __forceinline__ uint32_t pack_pair_to_u32<half>(float v1, float v2) {
    half2 packed = __floats2half2_rn(v1, v2);
    return *reinterpret_cast<uint32_t*>(&packed);
}
uint32_t u32_val = pack_pair_to_u32<OutT>(v1, v2);
```
## 新增加速比，并修改makefile
输出
```cpp
(cuda) ikko@dsw-607126-85f54bdf75-5lzlx:~/Learning-CUDA$ make nf4
=== [NF4] Compiling nf4/mainla.cu ===
TMPDIR=/home/ikko/Learning-CUDA/.tmp /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_80 nf4/mainla.cu -o nf4/mainla
=== [NF4] Running nf4/mainla ===
=== CUDA_VISIBLE_DEVICES=7 ===
CUDA_VISIBLE_DEVICES=7 ./nf4/mainla
SM count: 108, max active blocks/SM: 8, grid_x: 864
Kernel Time: 0.968471 ms
Effective Bandwidth (approx): 697.301 GB/s
Speedup vs bitsandbytes: 1.28384x (ref 1.24336 ms)
Bandwidth ratio vs bitsandbytes: 1.28383x (ref 543.14 GB/s)
Output dtype: bf16
Output written to nf4/data/output.bin
=== [NF4] Verifying MAE ===
CUDA_VISIBLE_DEVICES=7 python nf4/verify_mae.py
=== Starting Verification ===
Shape: 16384x16384, Blocksize: 64
------------------------------
MAE (Mean Absolute Error): 0.000017
Max Error:                 0.031250
------------------------------
✅ PASS: MAE (0.000017) is within threshold (0.01)
```
## 按题目要求回退线程粒度

题目要求的是 Packed Store：每个线程一次只处理两个 4-bit 索引，也就是读取 1 个 packed byte，算出 2 个 bf16 后，打包成 1 个 uint32_t，一次性写回全局内存。
而我前一版做的是：kernel 输入改成 const uint4*，每个线程先读 1 个 uint4，也就是 16 个 byte
16 个 byte 对应 32 个 4-bit 索引
线程内部虽然也调用了 pack_pair_to_u32<OutT>(v1, v2)
但那只是线程内部的中间步骤，最终是把 16 个 uint32 再组成 4 个 uint4 写回
也就是说，之前那版本质上是：
每线程处理 16 个 packed byte
每线程解码 32 个 4-bit 索引
每线程最终写 4 个 uint4
这不等于题目要求的：
每线程处理 1 个 packed byte，每线程解码 2 个 4-bit 索引，每线程最终写 1 个 uint32_t。所以这里重新改回严格按题目要求实现。
```cpp
template <typename OutT>
__global__ void nf4_decode_kernel(
    const uint8_t* __restrict__ packed_weights,
    const uint8_t* __restrict__ absmax_q,
    const half* __restrict__ absmax2,
    const half* __restrict__ code2,
    const float offset,
    OutT* __restrict__ output,
    int64_t num_elements,
    int blocksize,
    int group_size
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    int64_t total_bytes = (num_elements + 1) / 2;
    int64_t full_pair_bytes = num_elements / 2;

    __shared__ float s_LUT[16];
    if (threadIdx.x < 16) {
        s_LUT[threadIdx.x] = NF4_LUT[threadIdx.x];
    }
    __syncthreads();
    uint32_t* out_u32 = reinterpret_cast<uint32_t*>(output);

    for (int64_t byte_idx = tid; byte_idx < full_pair_bytes; byte_idx += stride) {
        uint8_t packed = packed_weights[byte_idx];
        int block_id = static_cast<int>(byte_idx / (blocksize / 2));
        int group_id = block_id / group_size;
        float real_absmax = (__half2float(absmax2[group_id]) * __half2float(code2[absmax_q[block_id]])) + offset;
        float v1 = s_LUT[packed >> 4] * real_absmax;
        float v2 = s_LUT[packed & 0x0F] * real_absmax;
        out_u32[byte_idx] = pack_pair_to_u32<OutT>(v1, v2);
    }

    if ((num_elements & 1) != 0) {
        int64_t tail_byte = total_bytes - 1;
        if (tid == 0) {
            uint8_t packed = packed_weights[tail_byte];
            int block_id = static_cast<int>(tail_byte / (blocksize / 2));
            int group_id = block_id / group_size;
            float real_absmax = (__half2float(absmax2[group_id]) * __half2float(code2[absmax_q[block_id]])) + offset;
            output[num_elements - 1] = float_to_out<OutT>(s_LUT[packed >> 4] * real_absmax);
        }
    }
}
```

同时 host 端的 grid 计算也要跟着改回按 total_bytes 来算，而不是按 total_words 算。因为现在一个线程对应一个 packed byte，不再是一个线程对应一个 uint4。

这版的好处是题意完全对齐，线程粒度、处理粒度、写回粒度三者一致。缺点也很明显，性能会比之前那个“每线程吞 16 个 byte”的版本低一些。

输出
```cpp
(cuda) ikko@dsw-607126-85f54bdf75-5lzlx:~/Learning-CUDA$ make nf4
=== [NF4] Compiling nf4/mainla.cu ===
TMPDIR=/home/ikko/Learning-CUDA/.tmp /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_80 nf4/mainla.cu -o nf4/mainla
=== [NF4] Running nf4/mainla ===
=== CUDA_VISIBLE_DEVICES=7 ===
CUDA_VISIBLE_DEVICES=7 ./nf4/mainla
SM count: 108, max active blocks/SM: 8, grid_x: 864
Kernel Time: 1.95401 ms
Effective Bandwidth (approx): 345.605 GB/s
Speedup vs bitsandbytes: 0.636311x (ref 1.24336 ms)
Bandwidth ratio vs bitsandbytes: 0.636309x (ref 543.14 GB/s)
Output dtype: bf16
Output written to nf4/data/output.bin
=== [NF4] Verifying MAE ===
CUDA_VISIBLE_DEVICES=7 python nf4/verify_mae.py
=== Starting Verification ===
Shape: 16384x16384, Blocksize: 64
------------------------------
MAE (Mean Absolute Error): 0.000017
Max Error:                 0.031250
------------------------------
✅ PASS: MAE (0.000017) is within threshold (0.01)
```


## nsys分析
```shell
nsys profile --stats=true --force-overwrite=true -o nf4_profile ./nf4/main
```
输出
```
Collecting data...
SM count: 108, max active blocks/SM: 8, grid_x: 864
Kernel Time: 1.85433 ms
Effective Bandwidth (approx): 364.183 GB/s
Speedup vs bitsandbytes: 0.670517x (ref 1.24336 ms)
Bandwidth ratio vs bitsandbytes: 0.670515x (ref 543.14 GB/s)
Output dtype: bf16
Output written to nf4/data/output_bf16.bin
Generating '/tmp/nsys-report-aac2.qdstrm'
[1/8] [========================100%] nsys_mainla_bf16.nsys-rep
[2/8] [========================100%] nsys_mainla_bf16.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /home/ikko/Learning-CUDA/nf4/nsys_mainla_bf16.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)           Name         
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------
     58.9       1684692553         24   70195523.0   73322110.5     320037  381847608   79085855.6  poll                  
     18.6        532343606       1622     328202.0      64362.5       1125   15994793     862809.4  ioctl                 
     11.3        322412024         31   10400387.9       1856.0       1022  322343688   57894226.3  fclose                
      8.1        232039676          1  232039676.0  232039676.0  232039676  232039676          0.0  writev                
      2.0         57202734        112     510738.7       2468.0       1002   56702430    5357453.0  fopen                 
      1.0         28063349         11    2551213.5       1931.0       1007   27199930    8178930.8  read                  
      0.1          1904289         43      44285.8      10845.0       6238    1058775     159640.8  mmap64                
      0.0           673049         10      67304.9      57184.0      21078     112731      29916.6  sem_timedwait         
      0.0           623082        118       5280.4       4386.0       1603      16509       3208.3  open64                
      0.0           306282          2     153141.0     153141.0     139640     166642      19093.3  pthread_create        
      0.0           201836         16      12614.8       5748.5       1002      88787      20920.5  mmap                  
      0.0            82243          1      82243.0      82243.0      82243      82243          0.0  pthread_cond_wait     
      0.0            77095         11       7008.6       7257.0       4709       9864       1769.3  write                 
      0.0            47414          8       5926.8       4498.0       2641      12043       3485.8  munmap                
      0.0            32173          3      10724.3      12344.0       6113      13716       4052.0  putc                  
      0.0            29046          1      29046.0      29046.0      29046      29046          0.0  fgets                 
      0.0            24664          5       4932.8       4357.0       2220       7422       2186.0  open                  
      0.0            20110          4       5027.5       4305.5       1200      10299       4042.1  fwrite                
      0.0            13228          3       4409.3       3261.0       1894       8073       3245.6  pipe2                 
      0.0            11141          2       5570.5       5570.5       5464       5677        150.6  socket                
      0.0             9397          2       4698.5       4698.5       1691       7706       4253.2  pthread_cond_broadcast
      0.0             7259          1       7259.0       7259.0       7259       7259          0.0  connect               
      0.0             6258          5       1251.6       1244.0       1080       1351        109.6  fcntl                 
      0.0             4160          1       4160.0       4160.0       4160       4160          0.0  fread                 
      0.0             2407          1       2407.0       2407.0       2407       2407          0.0  bind                  

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)           Name         
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------
     51.2        277706536          5   55541307.2    1658159.0       5925  272606419  121345975.1  cudaMalloc            
     34.5        187258524          1  187258524.0  187258524.0  187258524  187258524          0.0  cudaEventSynchronize  
     12.5         67776654          5   13555330.8    1488128.0      63639   51690255   22091691.8  cudaMemcpy            
      1.2          6327994          5    1265598.8    1131945.0      20003    2503567     933614.4  cudaFree              
      0.6          3056503          1    3056503.0    3056503.0    3056503    3056503          0.0  cudaDeviceSynchronize 
      0.1           578595        101       5728.7       4944.0       4228      36734       3821.5  cudaLaunchKernel      
      0.0            22548          2      11274.0      11274.0       5889      16659       7615.5  cudaEventRecord       
      0.0            17520          2       8760.0       8760.0        762      16758      11310.9  cudaEventDestroy      
      0.0             6073          2       3036.5       3036.5        780       5293       3191.2  cudaEventCreate       
      0.0             1196          1       1196.0       1196.0       1196       1196          0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------  --------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
    100.0        186128837        101  1842859.8  834756.0    812388   3455505    1258741.2  void nf4_decode_kernel<__nv_bfloat16>(const uint4 *, const unsigned char *, const __half *, const _…

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count   Avg (ns)    Med (ns)   Min (ns)  Max (ns)  StdDev (ns)           Operation          
 --------  ---------------  -----  ----------  ----------  --------  --------  -----------  ----------------------------
     78.2         51319384      1  51319384.0  51319384.0  51319384  51319384          0.0  [CUDA memcpy Device-to-Host]
     21.8         14291557      4   3572889.3    194241.0      3008  13900067    6887079.2  [CUDA memcpy Host-to-Device]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation          
 ----------  -----  --------  --------  --------  --------  -----------  ----------------------------
    536.871      1   536.871   536.871   536.871   536.871        0.000  [CUDA memcpy Device-to-Host]
    138.445      4    34.611     2.114     0.001   134.218       66.433  [CUDA memcpy Host-to-Device]
```

## maca
cd /data/Learning-CUDA && mxcc -O3 -std=c++17 nf4/mainla.maca -o nf4/mainla_maca
```cpp
cd /data/Learning-CUDA && mxcc -O3 -std=c++17 nf4/mainla.maca -o nf4/mainla_maca
nf4/mainla.maca:158:19: error: use of undeclared identifier 'macaMalloc'; did you mean 'mcMalloc'?
    RUNTIME_CHECK(macaMalloc(&d_packed, size_packed));
                  ^~~~~~~~~~
                  mcMalloc
nf4/../tester/utils.h:29:28: note: expanded from macro 'RUNTIME_CHECK'
    RUNTIME_ERR_TYPE err = call;                                               \
                           ^
/opt/maca/include/mcr/mc_runtime_api_template_wrapper.h:6:44: note: 'mcMalloc' declared here
template <class T> static inline mcError_t mcMalloc(T **devPtr, size_t size)
                                           ^
nf4/mainla.maca:159:19: error: use of undeclared identifier 'macaMalloc'; did you mean 'mcMalloc'?
    RUNTIME_CHECK(macaMalloc(&d_absmax_q, size_absmax_q));
                  ^~~~~~~~~~
                  mcMalloc
nf4/../tester/utils.h:29:28: note: expanded from macro 'RUNTIME_CHECK'
    RUNTIME_ERR_TYPE err = call;                                               \
                           ^
/opt/maca/include/mcr/mc_runtime_api_template_wrapper.h:6:44: note: 'mcMalloc' declared here
template <class T> static inline mcError_t mcMalloc(T **devPtr, size_t size)
                                           ^
nf4/mainla.maca:160:19: error: use of undeclared identifier 'macaMalloc'; did you mean 'mcMalloc'?
    RUNTIME_CHECK(macaMalloc(&d_absmax2, size_absmax2));
                  ^~~~~~~~~~
                  mcMalloc
nf4/../tester/utils.h:29:28: note: expanded from macro 'RUNTIME_CHECK'
    RUNTIME_ERR_TYPE err = call;                                               \
                           ^
/opt/maca/include/mcr/mc_runtime_api_template_wrapper.h:6:44: note: 'mcMalloc' declared here
template <class T> static inline mcError_t mcMalloc(T **devPtr, size_t size)
                                           ^
nf4/mainla.maca:161:19: error: use of undeclared identifier 'macaMalloc'; did you mean 'mcMalloc'?
    RUNTIME_CHECK(macaMalloc(&d_code2, size_code2));
                  ^~~~~~~~~~
                  mcMalloc
nf4/../tester/utils.h:29:28: note: expanded from macro 'RUNTIME_CHECK'
    RUNTIME_ERR_TYPE err = call;                                               \
                           ^
/opt/maca/include/mcr/mc_runtime_api_template_wrapper.h:6:44: note: 'mcMalloc' declared here
template <class T> static inline mcError_t mcMalloc(T **devPtr, size_t size)
                                           ^
nf4/mainla.maca:162:19: error: use of undeclared identifier 'macaMalloc'; did you mean 'mcMalloc'?
    RUNTIME_CHECK(macaMalloc(&d_output, static_cast<size_t>(num_elements) * sizeof(half)));
                  ^~~~~~~~~~
                  mcMalloc
nf4/../tester/utils.h:29:28: note: expanded from macro 'RUNTIME_CHECK'
    RUNTIME_ERR_TYPE err = call;                                               \
                           ^
/opt/maca/include/mcr/mc_runtime_api_template_wrapper.h:6:44: note: 'mcMalloc' declared here
template <class T> static inline mcError_t mcMalloc(T **devPtr, size_t size)
                                           ^
nf4/mainla.maca:164:70: error: use of undeclared identifier 'macaMemcpyHostToDevice'
    RUNTIME_CHECK(macaMemcpy(d_packed, h_packed.data(), size_packed, macaMemcpyHostToDevice));
                                                                     ^
nf4/mainla.maca:165:76: error: use of undeclared identifier 'macaMemcpyHostToDevice'
    RUNTIME_CHECK(macaMemcpy(d_absmax_q, h_absmax_q.data(), size_absmax_q, macaMemcpyHostToDevice));
                                                                           ^
nf4/mainla.maca:166:73: error: use of undeclared identifier 'macaMemcpyHostToDevice'
    RUNTIME_CHECK(macaMemcpy(d_absmax2, h_absmax2.data(), size_absmax2, macaMemcpyHostToDevice));
                                                                        ^
nf4/mainla.maca:167:67: error: use of undeclared identifier 'macaMemcpyHostToDevice'
    RUNTIME_CHECK(macaMemcpy(d_code2, h_code2.data(), size_code2, macaMemcpyHostToDevice));
                                                                  ^
nf4/mainla.maca:185:19: error: use of undeclared identifier 'macaGetLastError'; did you mean 'mcGetLastError'?
    RUNTIME_CHECK(macaGetLastError());
                  ^~~~~~~~~~~~~~~~
                  mcGetLastError
nf4/../tester/utils.h:29:28: note: expanded from macro 'RUNTIME_CHECK'
    RUNTIME_ERR_TYPE err = call;                                               \
                           ^
/opt/maca/include/mcr/mc_runtime_api.h:2311:11: note: 'mcGetLastError' declared here
mcError_t mcGetLastError(void);
          ^
nf4/mainla.maca:186:19: error: use of undeclared identifier 'macaDeviceSynchronize'; did you mean 'mcDeviceSynchronize'?
    RUNTIME_CHECK(macaDeviceSynchronize());
                  ^~~~~~~~~~~~~~~~~~~~~
                  mcDeviceSynchronize
nf4/../tester/utils.h:29:28: note: expanded from macro 'RUNTIME_CHECK'
    RUNTIME_ERR_TYPE err = call;                                               \
                           ^
/opt/maca/include/mcr/mc_runtime_api.h:570:11: note: 'mcDeviceSynchronize' declared here
mcError_t mcDeviceSynchronize(void);
          ^
nf4/mainla.maca:194:19: error: use of undeclared identifier 'macaGetLastError'; did you mean 'mcGetLastError'?
    RUNTIME_CHECK(macaGetLastError());
                  ^~~~~~~~~~~~~~~~
                  mcGetLastError
nf4/../tester/utils.h:29:28: note: expanded from macro 'RUNTIME_CHECK'
    RUNTIME_ERR_TYPE err = call;                                               \
                           ^
/opt/maca/include/mcr/mc_runtime_api.h:2311:11: note: 'mcGetLastError' declared here
mcError_t mcGetLastError(void);
          ^
nf4/mainla.maca:195:19: error: use of undeclared identifier 'macaDeviceSynchronize'; did you mean 'mcDeviceSynchronize'?
    RUNTIME_CHECK(macaDeviceSynchronize());
                  ^~~~~~~~~~~~~~~~~~~~~
                  mcDeviceSynchronize
nf4/../tester/utils.h:29:28: note: expanded from macro 'RUNTIME_CHECK'
    RUNTIME_ERR_TYPE err = call;                                               \
                           ^
/opt/maca/include/mcr/mc_runtime_api.h:570:11: note: 'mcDeviceSynchronize' declared here
mcError_t mcDeviceSynchronize(void);
          ^
nf4/mainla.maca:201:107: error: use of undeclared identifier 'macaMemcpyDeviceToHost'
    RUNTIME_CHECK(macaMemcpy(h_output.data(), d_output, static_cast<size_t>(num_elements) * sizeof(half), macaMemcpyDeviceToHost));
                                                                                                          ^
nf4/mainla.maca:217:19: error: use of undeclared identifier 'macaFree'; did you mean 'mcFree'?
    RUNTIME_CHECK(macaFree(d_packed));
                  ^~~~~~~~
                  mcFree
nf4/../tester/utils.h:29:28: note: expanded from macro 'RUNTIME_CHECK'
    RUNTIME_ERR_TYPE err = call;                                               \
                           ^
/opt/maca/include/mcr/mc_runtime_api.h:2609:11: note: 'mcFree' declared here
mcError_t mcFree(void *ptr);
          ^
nf4/mainla.maca:218:19: error: use of undeclared identifier 'macaFree'; did you mean 'mcFree'?
    RUNTIME_CHECK(macaFree(d_absmax_q));
                  ^~~~~~~~
                  mcFree
nf4/../tester/utils.h:29:28: note: expanded from macro 'RUNTIME_CHECK'
    RUNTIME_ERR_TYPE err = call;                                               \
                           ^
/opt/maca/include/mcr/mc_runtime_api.h:2609:11: note: 'mcFree' declared here
mcError_t mcFree(void *ptr);
          ^
nf4/mainla.maca:219:19: error: use of undeclared identifier 'macaFree'; did you mean 'mcFree'?
    RUNTIME_CHECK(macaFree(d_absmax2));
                  ^~~~~~~~
                  mcFree
nf4/../tester/utils.h:29:28: note: expanded from macro 'RUNTIME_CHECK'
    RUNTIME_ERR_TYPE err = call;                                               \
                           ^
/opt/maca/include/mcr/mc_runtime_api.h:2609:11: note: 'mcFree' declared here
mcError_t mcFree(void *ptr);
          ^
nf4/mainla.maca:220:19: error: use of undeclared identifier 'macaFree'; did you mean 'mcFree'?
    RUNTIME_CHECK(macaFree(d_code2));
                  ^~~~~~~~
                  mcFree
nf4/../tester/utils.h:29:28: note: expanded from macro 'RUNTIME_CHECK'
    RUNTIME_ERR_TYPE err = call;                                               \
                           ^
/opt/maca/include/mcr/mc_runtime_api.h:2609:11: note: 'mcFree' declared here
mcError_t mcFree(void *ptr);
          ^
nf4/mainla.maca:221:19: error: use of undeclared identifier 'macaFree'; did you mean 'mcFree'?
    RUNTIME_CHECK(macaFree(d_output));
                  ^~~~~~~~
                  mcFree
nf4/../tester/utils.h:29:28: note: expanded from macro 'RUNTIME_CHECK'
    RUNTIME_ERR_TYPE err = call;                                               \
                           ^
/opt/maca/include/mcr/mc_runtime_api.h:2609:11: note: 'mcFree' declared here
mcError_t mcFree(void *ptr);
          ^
19 errors generated when compiling for host.
```
原来沐曦用的是mc而不是maca


## ncu

4090上ncu的结果
```shell
sudo ncu \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --launch-skip 1 \
  --launch-count 1 \
  ./nf4/mainla bf16
```
拿到的关键信息大致是：
	•	DRAM Throughput ≈ 83%
	•	Memory ≈ 83%
	•	Compute (SM) ≈ 38% ~ 42%
	•	Memory Throughput ≈ 816 ~ 824 GB/s
	•	L2 Hit Rate ≈ 79.84%
说明 1.	这个 kernel 已经明显是 memory-bound；
	2.	不是算力没吃满，而是 DRAM 带宽已经被压得比较高；
	3.	继续优化如果不动访存主路径，很难有大提升。

```shell
sudo ncu \
  --section SchedulerStats \
  --section WarpStateStats \
  --launch-skip 1 \
  --launch-count 1 \
  ./nf4/mainla bf16
==PROF== Connected to process 810279 (/home/xjy/Learning-CUDA/nf4/mainla)
SM count: 128, max active blocks/SM: 6, grid_x: 768
==PROF== Profiling "nf4_decode_kernel" - 0 (1/1): 0%....50%....100% - 8 passes
Kernel Time: 2.01215 ms
Effective Bandwidth (approx): 335.62 GB/s
Speedup vs bitsandbytes: 0.617927x (ref 1.24336 ms)
Bandwidth ratio vs bitsandbytes: 0.617925x (ref 543.14 GB/s)
Output dtype: bf16
Output written to nf4/data/output_bf16.bin
==PROF== Disconnected from process 810279
[810279] mainla@127.0.0.1
  void nf4_decode_kernel<__nv_bfloat16>(const unsigned char *, const unsigned char *, const __half *, const __half *, float, T1 *, long, int, int), 2026-Mar-10 14:16:43, Context 1, Stream 7
    Section: Scheduler Statistics
    ---------------------------------------------------------------------- --------------- ------------------------------
    One or More Eligible                                                                 %                          42.23
    Issued Warp Per Scheduler                                                                                        0.42
    No Eligible                                                                          %                          57.77
    Active Warps Per Scheduler                                                        warp                          12.08
    Eligible Warps Per Scheduler                                                      warp                           1.11
    ---------------------------------------------------------------------- --------------- ------------------------------
    WRN   Every scheduler is capable of issuing one instruction per cycle, but for this kernel each scheduler only      
          issues an instruction every 2.4 cycles. This might leave hardware resources underutilized and may lead to     
          less optimal performance. Out of the maximum of 12 warps per scheduler, this kernel allocates an average of   
          12.08 active warps per scheduler, but only an average of 1.11 warps were eligible per cycle. Eligible warps   
          are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible   
          warp results in no instruction being issued and the issue slot remains unused. To increase the number of      
          eligible warps, avoid possible load imbalances due to highly different execution durations per warp.          
          Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too.            

    Section: Warp State Statistics
    ---------------------------------------------------------------------- --------------- ------------------------------
    Warp Cycles Per Issued Instruction                                               cycle                          28.61
    Warp Cycles Per Executed Instruction                                             cycle                          28.61
    Avg. Active Threads Per Warp                                                                                    19.67
    Avg. Not Predicated Off Threads Per Warp                                                                        18.16
    ---------------------------------------------------------------------- --------------- ------------------------------
    WRN   On average, each warp of this kernel spends 19.0 cycles being stalled waiting for a scoreboard dependency on  
          a L1TEX (local, global, surface, texture, rtcore) operation. This represents about 66.5% of the total         
          average of 28.6 cycles between issuing two instructions. To reduce the number of cycles waiting on L1TEX      
          data accesses verify the memory access patterns are optimal for the target architecture, attempt to increase  
          cache hit rates by increasing data locality or by changing the cache configuration, and consider moving       
          frequently used data to registers and to shared memory.                                                       
    ----- --------------------------------------------------------------------------------------------------------------
    INF   Check the Source Counters section for the top stall locations in your source based on sampling data. The      
          Kernel Profiling Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sampling) provides   
          more details on each stall reason.                                                                            
    ----- --------------------------------------------------------------------------------------------------------------
    WRN   Instructions are executed in warps, which are groups of 32 threads. Optimal instruction throughput is         
          achieved if all 32 threads of a warp execute the same instruction. The chosen launch configuration, early     
          thread completion, and divergent flow control can significantly lower the number of active threads in a warp  
          per cycle. This kernel achieves an average of 19.7 threads being active per cycle. This is further reduced    
          to 18.2 threads per warp due to predication. The compiler may use predication to avoid an actual branch.      
          Instead, all instructions are scheduled, but a per-thread condition code or predicate controls which threads  
          execute the instructions. Try to avoid different execution paths within a warp when possible. In addition,    
          ensure your kernel makes use of Independent Thread Scheduling, which allows a warp to reconverge after a      
          data-dependent conditional block by explicitly calling __syncwarp().
```
不是抽象地 memory-bound，而是 大量 warp 在等 L1TEX 路径上的 load 返回；调度器手上虽然挂了很多 warp，但真正能随时发射的 warp 很少；所以 issue slot 经常空着。
怀疑：1.不只是主数据流 packed_weights 在拖；
2.absmax_q / absmax2 / code2 这些访问也可能形成依赖链；
3.另外，热路径里的整数计算（特别是除法）可能在吃性能。
### blocksize,group_size
printf("group_size: %d,block_size: %d",group_size,blocksize);
得到block_size: 64；group_size:256
```cpp
float real_absmax = 0.0f;
        if (lane == 0) {
            uint8_t qa = absmax_q[block_id];
            real_absmax = (__half2float(absmax2[group_id]) * __half2float(code2[qa])) + offset;
        }
        real_absmax = __shfl_sync(warp_mask, real_absmax, 0);
```
没啥变化
可能code2 / absmax2 这种小数据本来就可能已经被 cache 命中；新加的 if (lane == 0) 和 __shfl_sync 也有额外开销。

综上
综合几轮 NCU，我目前对这个 kernel 的判断是：

已经确认的事实
	1.	它是明显的 memory-bound kernel；
	2.	DRAM 带宽已经吃到比较高；
	3.	warp 大量 stall 在 L1TEX scoreboard dependency 上；
	4.	shared LUT 在我的测试里更快，应该保留；
	5.	real_absmax 的 warp 广播不是主收益点；
	6.	热路径里的运行时除法是一个更值得优先处理的问题。

热路径的主要成本大概来自：
主成本
	•	packed_weights[byte_idx] 的 load
	•	out_u32[byte_idx] 的 store
次成本
	•	absmax_q[block_id]
	•	absmax2[group_id]
	•	code2[qa]
可能被低估的成本
	•	byte_idx / (blocksize / 2)
	•	block_id / group_size

## 由于A100显存占用
最终在4090测试。
结果：rows=16384, cols=16384, blocksize=64
```shell
./nf4/mainla 
SM count: 128, max active blocks/SM: 6, grid_x: 768
Kernel Time: 0.790252 ms
Effective Bandwidth (approx): 854.559 GB/s
Speedup vs bitsandbytes: 1.57337x (ref 1.24336 ms)
Bandwidth ratio vs bitsandbytes: 1.57337x (ref 543.14 GB/s)
Output dtype: bf16
Output written to nf4/data/output.bin
```
rows=24576, cols=24576, blocksize=64
```shell
./nf4/mainla 
SM count: 128, max active blocks/SM: 6, grid_x: 768
Kernel Time: 1.77564 ms
Effective Bandwidth (approx): 855.727 GB/s
Speedup vs bitsandbytes: 0.700233x (ref 1.24336 ms)
Bandwidth ratio vs bitsandbytes: 1.57552x (ref 543.14 GB/s)
Output dtype: bf16
Output written to nf4/data/output.bin
```