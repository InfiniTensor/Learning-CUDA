// =============================================================================
//  platform.hpp -- 设备编译器探测与跨平台差异的集中抽象
//
//  本项目的原始开发环境是 Windows + nvcc。要把同一份源码交给国产平台编译器
//  (摩尔 mcc / 天数 clang++ -x cuda / 沐曦 mxcc) 编译, 需要感知平台的只有下面
//  五处, 全部集中在本文件:
//
//    1. 判断「当前是否被设备编译器编译」      -> LPQ_CUDA
//    2. 判断「当前是否处于设备编译遍」        -> LPQ_DEVICE_PASS
//    3. 低精度类型头文件 (见 vec_io.cuh)      -> LPQ_FP16_HEADER / LPQ_BF16_HEADER
//    4. warp 内数据交换                       -> LPQ_SHFL_XOR
//    5. 浮点原子最大                          -> lpq_atomic_max_f32
//
//  以上都是「同名替换」: 在英伟达平台上展开的结果与本项目最初使用的写法完全
//  一致, 因此不会改变已经验证过的数值行为。
//
//  本文件不包含任何头文件, 可在主机 / 设备两侧安全包含。
//
//  ---------------------------------------------------------------------------
//  命令行开关(某个平台的能力未被自动识别时使用)
//  ---------------------------------------------------------------------------
//    -DLPQ_CUDA=1               强制认定处于设备编译
//    -DLPQ_DEVICE_PASS=1        强制认定处于设备编译遍
//    -DLPQ_FP16_HEADER=<h>      改用 <h> 作为半精度类型头
//    -DLPQ_BF16_HEADER=<h>      改用 <h> 作为 bfloat16 类型头
//    -DLPQ_SHFL_NO_SYNC=1       仅用于只提供 CUDA 9 之前老接口 __shfl_xor 的
//                               旧工具链. 注意 CUDA 9 起 nvcc 已移除 __shfl_xor,
//                               因此在英伟达平台上**不要**打开这个开关.
//    -DLPQ_NO_ATOMIC_MAX_INT=1  该平台没有 atomicMax(int*), 改用 CAS 循环
//    -DLPQ_NO_ATOMIC_ADD_F64=1  该平台没有 atomicAdd(double*), 改用 CAS 循环
//
//  tests/test_portability.cu 会在设备上把上述宏逐一打印出来, 上机时先跑它.
// =============================================================================
#pragma once

// -----------------------------------------------------------------------------
//  1. 设备编译器探测
//     nvcc / 天数 clang++(-x cuda) / 沐曦 mxcc 都定义 __CUDACC__(为兼容 CUDA
//     而保留); 摩尔线程 mcc 定义 __MUSACC__. 两者都认, 再留一个显式覆盖入口.
// -----------------------------------------------------------------------------
#ifndef LPQ_CUDA
#if defined(__CUDACC__) || defined(__MUSACC__) || defined(__MUSA__) || defined(__MACA__)
#define LPQ_CUDA 1
#else
#define LPQ_CUDA 0
#endif
#endif

// -----------------------------------------------------------------------------
//  2. 设备编译遍探测
//     一次 nvcc/mcc 编译会走两遍: 主机遍 + 设备遍. __CUDA_ARCH__ 只在设备遍
//     被定义(nvcc), 摩尔对应 __MUSA_ARCH__, 沐曦对应 __MACA_ARCH__.
// -----------------------------------------------------------------------------
#ifndef LPQ_DEVICE_PASS
#if defined(__CUDA_ARCH__) || defined(__MUSA_ARCH__) || defined(__MACA_ARCH__)
#define LPQ_DEVICE_PASS 1
#else
#define LPQ_DEVICE_PASS 0
#endif
#endif

// -----------------------------------------------------------------------------
//  函数限定符
//    LPQ_HD : 主机与设备两侧都需要可见(两侧共享的纯计算函数)
//    LPQ_DEV: 仅设备侧
// -----------------------------------------------------------------------------
#if LPQ_CUDA
#define LPQ_HD __host__ __device__ __forceinline__
#define LPQ_DEV __device__ __forceinline__
#else
#define LPQ_HD inline
#define LPQ_DEV inline
#endif

// -----------------------------------------------------------------------------
//  4. warp 内交换
//     用法: LPQ_SHFL_XOR(v, off, width)
//     width 必须是 2 的幂且 <= 32, 表示参与交换的子组宽度.
// -----------------------------------------------------------------------------
#if LPQ_CUDA
#if defined(LPQ_SHFL_NO_SYNC)
// 老接口: 不带 active mask, 但同样接受 width 参数. 只在 CUDA 9 之前的工具链上存在.
#define LPQ_SHFL_XOR(v, off, width) __shfl_xor((v), (off), (width))
#else
#define LPQ_SHFL_XOR(v, off, width) __shfl_xor_sync(0xFFFFFFFFu, (v), (off), (width))
#endif
#else
#define LPQ_SHFL_XOR(v, off, width) (v)
#endif

// -----------------------------------------------------------------------------
//  5. 浮点原子最大
//     CUDA 没有 atomicMax(float*), 惯用做法是把 float 按位重解释成 int 再取最大
//     -- 这对非负数是正确的: IEEE-754 的正浮点位模式按 uint32 比较与按数值比较
//     同序. 本函数因此**只允许用于非负值**(项目里只用它求绝对值最大, 满足前提).
// -----------------------------------------------------------------------------
namespace lpq {

LPQ_DEV void lpq_atomic_max_f32(float* addr, float val) {
#if defined(LPQ_NO_ATOMIC_MAX_INT)
    // 回退实现: 比较-交换循环. 慢但对任何指令集都能工作.
    int* const ai = reinterpret_cast<int*>(addr);
    int old = *ai;
    while (__float_as_int(val) > old) {
        const int prev = atomicCAS(ai, old, __float_as_int(val));
        if (prev == old)
            break;
        old = prev;
    }
#else
    atomicMax(reinterpret_cast<int*>(addr), __float_as_int(val));
#endif
}

// -----------------------------------------------------------------------------
//  6. 双精度原子加
//     atomicAdd(double*) 自 sm_60 起才是硬件指令 (更早的架构由 nvcc 生成 CAS
//     序列). 部分国产平台没有实现它, 因此这里提供回退.
//     本项目只在误差统计 kernel 里、每 block 结束时调用 4 次, 回退带来的额外
//     开销可以忽略.
// -----------------------------------------------------------------------------
LPQ_DEV void lpq_atomic_add_f64(double* addr, double val) {
#if defined(LPQ_NO_ATOMIC_ADD_F64)
    union F64Bits {
        double d;
        unsigned long long u;
    };
    unsigned long long* const ai = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old;
    F64Bits cur;
    cur.u = *ai;
    old = cur.u;
    for (;;) {
        const unsigned long long assumed = old;
        F64Bits want;
        want.d = cur.d + val;
        old = atomicCAS(ai, assumed, want.u);
        if (old == assumed)
            return;
        cur.u = old;
    }
#else
    atomicAdd(addr, val);
#endif
}

}  // namespace lpq
