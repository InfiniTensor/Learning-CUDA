#pragma once
#include <cfloat>
#include <cmath>
#include <cstdint>
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
namespace lp {
// 同一标量契约用于 CPU/GPU；tests 中另用枚举距离参考验证。
// f=0 返回 126（E4M3 最大有限编码），f=1 返回 123（E5M2 最大有限编码），否则返回 7（E2M1 最大有限编码 +6 对应编码 7）。
HD inline int maxcode(int f) {
    if(f==0) return 126;
    else if(f==1) return 123;
    else return 7;
}

//把编码 c 按格式 f 解码成 double。
HD inline double decode(unsigned c, int f) {
    //根据格式设置尾数位数 mb 和指数偏置 bias：
    // f=0：E4M3，mb=3，bias=7
    // f=1：E5M2，mb=2，bias=15
    // f=2：E2M1，mb=1，bias=1
    int mb,bias;
    if(f==0){
        mb=3;
        bias=7;
    }
    else if(f==1){
        mb=2;
        bias=15;
    }
    else{
        mb=1;
        bias=1;
    }
    //确定符号位掩码 sign：f=2 是 4 位格式，符号位为 8；否则是 8 位格式，符号位为 128。
    // a = c & (sign - 1)：去掉符号位后的绝对值部分。
    // e = a >> mb：提取指数位。
    // m = a & ((1 << mb) - 1)：提取尾数位。
    unsigned sign = f == 2 ? 8 : 128;
    unsigned a = c & (sign - 1);
    unsigned e = a >> mb;
    unsigned m = a & ((1 << mb) - 1);

    double v;
    //如果格式 0（E4M3）且绝对值部分为 127，这是 NaN 编码。
    if (f == 0 && a == 127)
        v = NAN;
    //否则如果格式 1（E5M2）且指数为 31，这是 Inf/NaN 编码。
    else if (f == 1 && e == 31)
        //尾数非零则为 NaN，否则为无穷大。
        v = m ? NAN : INFINITY;
    else
        //计算浮点值：
        //如果 e 非零（正规数），有效数字为 1 + m / 2^mb
        //如果 e 为零（次正规数），有效数字为 m / 2^mb
        //然后通过 ldexp 乘以 2 的幂。
        v = ldexp(e ? 1.0 + double(m) / (1 << mb) : double(m) / (1 << mb),
                  e ? int(e) - bias : 1 - bias);
    return c & sign ? -v : v;
}
HD inline double uniform(uint64_t seed, uint64_t i) {
    uint64_t z = seed ^ (i + 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return double(z >> 11) * 0x1p-53;
}
//把 double 编码为指定格式的编码。   参数：x：输入值   f：格式编号    stochastic：是否使用随机舍入    seed：随机种子
HD inline uint8_t encode(double x, int f, bool stochastic = false, uint64_t seed = 42,
                         uint64_t i = 0) {
    //如果输入是 NaN。返回对应格式的 NaN 编码：f=0 返回 127，f=1 返回 126，f=2 返回 0。
    if (std::isnan(x)){
        if(f==0) return 127;
        else if(f==1) return 126;
        else return 0;
    }
    //确定符号位：如果 x 为负，f=2 时符号位为 8，否则为 128；如果为正，符号位为 0。
    //std::signbit(x) 用来判断一个浮点数的符号位是不是负的，返回 bool：
    unsigned s = std::signbit(x) ? (f == 2 ? 8 : 128) : 0;
    //取绝对值，后续只处理正值。
    x = fabs(x);
    //获取该格式最大有限编码。
    int top = maxcode(f);
    //饱和到最大有限编码，并加上符号位。
    if (x >= decode(top, f))
        return s | top;
    // 单调正编码二分寻找相邻值，不用噪声近似随机舍入。
    int lo = 0, hi = top;
    while (hi - lo > 1) {
        int mid = (hi + lo) / 2;
        if (decode(mid, f) <= x)
            lo = mid;
        else
            hi = mid;
    }
    //得到相邻的两个可表示值 a 和 b，且 a <= x <= b。
    double a = decode(lo, f), b = decode(hi, f);
    //如果启用随机舍入，则用随机数决定是否向上舍入，概率为 (x-a)/(b-a)。
    //否则四舍五入：如果 x 离 b 更近，或距离相等且 lo 为奇数（保证舍入到偶数编码），则向上。
    bool up = stochastic ? uniform(seed, i) < (x - a) / (b - a)
                         : (x - a > b - x || (x - a == b - x && (lo & 1)));
    //返回符号位加上选中的编码。
    return s | (up ? hi : lo);
}

//计算 MX 格式的共享缩放因子，返回 8 位指数。
HD inline uint8_t mxscale(double a, int f) {
    //如果输入为 0。返回 127，表示 2^0 = 1 的缩放。
    if (a == 0)
        return 127;
    //计算 a 除以该格式最大可表示值。
    double r = a / decode(maxcode(f), f);
    //声明指数变量
    int e;
    //将 r 分解为 m * 2^e，其中 m 在 [0.5, 1) 之间。
    double m = frexp(r, &e);
    //如果尾数正好是 0.5。指数减 1，相当于把 0.5 * 2^e 调整为 1.0 * 2^(e-1)，便于后续舍入到最近的 2 的幂。
    if (m == 0.5)
        --e;
    if (e < -127)
        e = -127;
    if (e > 127)
        e = 127;
    //返回偏置 127 的 8 位指数编码。
    return uint8_t(e + 127);
}

//定义 global_scale，计算 NVFP4 的全局 FP32 缩放。
//x ≈ q_E2M1​×s_(block,E4M3)​×s_(global,FP32)
// 其中：q_E2M1 是 4 bit 元素，最大绝对值 6
//      s_(block,E4M3) 是每 16 个元素共享的块缩放，E4M3 最大有限值 448
//      s_(global,FP32) 就是这里的 global_scale
HD inline float global_scale(double a) {
    //如果 a 为 0 返回 1.0f；否则返回 a/2688 和 2^-126 的较大值，转为 float。
    //其中 2688 = 448 * 6，因为 NVFP4 块缩放用 E4M3 最大 448，元素 E2M1 最大 6。
    return a == 0 ? 1.0f : float(fmax(a / 2688.0, 0x1p-126));
}

//定义 nvscale，计算 NVFP4 的块缩放。
//a：当前微块（16 个元素）中的最大绝对值（原始未缩放数据）。
//g：整个张量的全局缩放因子（FP32）。
HD inline uint8_t nvscale(double a, float g) {
    if (a == 0)
        //返回 56，对应 E4M3 中的 1.0。
        return 56;
    //原始数据先除以全局缩放 g 得到 x_scaled = x / g，微块最大值变为 a / g。块缩放定义为 max(|x_scaled|) / 6，
    //s_(block,E4M3)=x/(q_E2M1 * s_(global,FP32))
    uint8_t s = encode(a / (6.0 * double(g)), 0);
    return s ? s : 1;
}

//定义 scale_value，根据缩放编码和格式还原缩放值。
//MXFP8还原共享缩放因子，NVFP4还原 块缩放*全局FP32缩放
HD inline double scale_value(uint8_t s, int f, float g) {
    //如果是 f=2（NVFP4），返回 decode(s, 0) * g；否则（MX）返回 2^(s-127)。
    return f == 2 ? decode(s, 0) * double(g) : ldexp(1.0, int(s) - 127);
}

// 定义 bits，将 float 的位模式转为 uint32_t
HD inline uint32_t bits(float x) {
    union {
        float f;
        uint32_t u;
    } v;
    //把 x 的 IEEE 754 单精度浮点位模式写进这 4 字节。
    v.f = x;
    //把这同样的 4 字节按 uint32_t 读出来。
    return v.u;
}

//定义 frombits，将 uint32_t 位模式转为 float。
HD inline float frombits(uint32_t x) {
    union {
        float f;
        uint32_t u;
    } v;
    v.u = x;
    return v.f;
}

// 格式	 符号	指数	尾数	偏置
// FP32	 1	    8	   23	 127
// FP16	 1	    5	   10	 15

//偏置 bias = 2^(e-1) - 1。（e为指数的位数）     
//正规数   1≤E≤2^e−2    (−1)^S×2^(E−bias)×(1.M)_2
//非正规数  E=0         (−1)^S×2^(1−bias)×(0.M)_2

//    零：E = 0, M = 0，值 = ±0。
//    无穷 Inf：E 全 1，M = 0，值 = ±∞。
//    NaN：E 全 1，M ≠ 0，表示非数。

//定义 half_bits，将 FP32 转为 FP16 位模式。
HD inline uint16_t half_bits(float x) {
    // u = bits(x)：取 FP32 的 32 位原样。
    // s = (u >> 16) & 0x8000：把符号位从第 31 位移到第 15 位，作为 FP16 的符号位。
    // a = u & 0x7fffffff：去掉符号位后的绝对值。
    uint32_t u = bits(x), s = (u >> 16) & 0x8000, a = u & 0x7fffffff;
    // 0x7f800000 是 FP32 的指数全 1、尾数为 0，即 Inf。
    // a == 0x7f800000：Inf → FP16 的 Inf 是 0x7c00。
    // a > 0x7f800000：NaN → FP16 的 NaN 是 0x7e00（quiet NaN）。
    if (a >= 0x7f800000)
        return s | (a == 0x7f800000 ? 0x7c00 : 0x7e00);
    //e：FP32 的无偏指数。
    //m：把隐含的 1 补回来，得到 24 位有效数字（范围 [2^23, 2^24)）。
    int e = int(a >> 23) - 127;
    uint32_t m = (a & 0x7fffff) | 0x800000;
    //FP16 正规数最大指数是 15 → e > 15 溢出为 Inf。
    //FP16 最小非正规数是 2^-24 → e < -25 直接下溢为 0。
    if (e > 15)
        return s | 0x7c00;
    if (e < -25)
        return s;
    //FP16 最小正规数的实际指数是：1−15=−14
    //所以：当 e >= -14 时，FP32 的数可以表示为 FP16 的正规数。
    //     当 e < -14 时，FP32 的数太小，只能表示为 FP16 的非正规数。

    //正规数（e >= -14）：FP32 有 23 位尾数，FP16 只有 10 位，所以右移 13 位，丢掉多余位。
    //非正规数（e < -14）：还需要额外右移，把有效数字对齐到 2^-14 的倍数。公式 shift = -e - 1。
    //例如 e = -15 → shift = 14，把 24 位有效数字右移 14 位，得到 10 位尾数。
    int shift = e < -14 ? -e - 1 : 13;
    //q：截断后的值。
    //rem：被丢掉的低位。
    //mid：正好一半的位置。
    uint32_t q = m >> shift; 
    uint32_t rem = m & ((1u << shift) - 1); 
    uint32_t mid = 1u << (shift - 1);
    // 规则：rem > mid → 向上入。
    //      rem == mid 且 q 是奇数 → 向上入，使结果变成偶数。
    //      否则截断。
    // 这就是标准的 round-half-to-even，避免统计偏差

    //(rem > mid)：大于一半，结果为 true（1），进位。
    //(rem == mid && (q & 1))：恰好一半，且 q 是奇数（q & 1 == 1），进位。
    //其他情况条件为 false（0），不进位。
    q += (rem > mid || (rem == mid && (q & 1)));
    //当 e < -14 时，数值太小，FP16 只能用非正规数表示。
    //FP16 非正规数的位布局是：S | 00000 | MMMMMMMMMM
    if (e < -14)
        return s | q;
    // q = m >> 13 的范围是 [2^10, 2^11)，也就是 11 位，最高位是隐含的 1。
    // FP16 的指数域应该放 e + 15，左移 10 位就是 (e + 15) * 1024。
    // 但 q 里多了一个隐含的 2^10 = 1024，所以用 (e + 14) * 1024 + q 正好抵消：
    // (e+14)×1024+q=(e+15)×1024+(q−1024)
    // 其中 q - 1024 就是真正的 10 位尾数。
    return s | uint32_t((e + 14) * 1024 + q);
}

//定义 half_value，将 FP16 位模式转为 FP32。
HD inline float half_value(uint16_t h) {
    //提取 FP16 指数和尾数。
    int e = (h >> 10) & 31;
    int m = h & 1023;
    //如果指数为 31（Inf/NaN）。
    float v = e == 31
                  //尾数非零为 NaN，否则 Inf
                  ? (m ? NAN : INFINITY)
                  //否则计算值：正规数：(1 + m/1024) * 2^(e-15)
                  //        次正规数：(m/1024) * 2^-14
                  : float(ldexp(e ? 1.0 + double(m) / 1024 : double(m) / 1024, e ? e - 15 : -14));
    //根据符号位返回正负。
    return h & 32768 ? -v : v;
}

//定义 bf_bits，将 FP32 转为 BF16 位模式。
HD inline uint16_t bf_bits(float x) {
    uint32_t u = bits(x);
    // 0x7f800000 是 FP32 的指数全 1、尾数为 0，即 Inf。
    // == 0x7f800000：Inf
    //  > 0x7f800000：NaN
    if ((u & 0x7fffffff) > 0x7f800000)
        //返回 BF16 NaN，保留符号和指数，设置尾数最高位确保是 NaN。
        // | 64（即 0x40，二进制 0100 0000）强制把尾数的最高位（第 6 位）置 1，确保结果仍是 NaN
        return uint16_t((u >> 16) | 64);
    //u + 0x7fff：0x7fff 是 0x8000 - 1。低 16 位的中点是 0x8000。
    //加上 0x7fff 后，如果低 16 位 ≥ 0x8000，就会向高 16 位进位；如果低 16 位 < 0x8000，则不会进位。

    //((u >> 16) & 1)：取高 16 位的最低位（即 BF16 尾数的最低位）。
    // L < 0x8000	舍去
    // L > 0x8000	进位
    // L == 0x8000	看BF16尾数的最低位奇偶：奇数则进位到，偶数则保持

    //如果低16位刚好等于0x8000,然后BF16尾数的最低位是1(0x0001)
    //0x8000+0x7fff+0x0001=0x8000+0x8000=0x10000=0x00010000
    //u+0x00010000,由于BF16尾数的最低位是1，1+1就进位到0
    return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

//定义 finite_float，将 double 转为有限范围的 float。
HD inline float finite_float(double x) {
    //如果 x 是 NaN（非数）或 Inf（无穷大），直接把它转成 float 返回。
    if (std::isnan(x) || std::isinf(x))
        return float(x);
    if(x>double(FLT_MAX))
        return FLT_MAX;
    if(x<-double(FLT_MAX))
        return -FLT_MAX;
    return float(x);
}

//定义 output_bits，根据 dtype 输出 FP32/FP16/BF16 位模式。
HD inline uint32_t output_bits(double x, int dtype) {
    float v = finite_float(x);
    return dtype == 0 ? bits(v) : dtype == 1 ? half_bits(v) : bf_bits(v);
}

//定义 input_value，从内存读取输入值。
HD inline float input_value(const uint8_t *p, uint64_t i, int dtype) {
    //dtype=0 时按 float 读取。
    //否则按 uint16_t 读取并将这个 FP16 编码解析成 FP32 的 float
    return dtype == 0 ? reinterpret_cast<const float *>(p)[i]
                      : half_value(reinterpret_cast<const uint16_t *>(p)[i]);
}

//定义 store_value，将值写入内存。
HD inline void store_value(uint8_t *p, uint64_t i, int dtype, uint32_t v) {
    if (dtype == 0)
        reinterpret_cast<uint32_t *>(p)[i] = v;
    else
        reinterpret_cast<uint16_t *>(p)[i] = uint16_t(v);
}
} // namespace lp


namespace lp {
// 所有私有元素的有限值都能被 FP32 精确表示；直接构造位域避免反复调用 ldexp。
//定义 decode_fast，快速解码。它可以把三种不同格式的浮点位模式（E4M3、E5M2、E2M1）转换成 double 返回。
HD inline double decode_fast(unsigned c, int f) {
    //同 decode，设置尾数位数和偏置。
    int mb = f == 0 ? 3 : f == 1 ? 2 : 1; 
    int bias = f == 0 ? 7 : f == 1 ? 15 : 1;
    // sign：符号位掩码。4 位格式（E2M1）符号位在第 3 位，掩码 8；8 位格式符号位在第 7 位，掩码 128。
    // a = c & (sign - 1)：去掉符号位，得到绝对值部分的位模式。
    // e = a >> mb：提取指数位。
    // m = a & ((1 << mb) - 1)：提取尾数位。
    unsigned sign = f == 2 ? 8 : 128; 
    unsigned a = c & (sign - 1); 
    unsigned e = a >> mb;
    unsigned m = a & ((1 << mb) - 1);
    //声明 float 结果。
    float v;
    // E4M3：a == 127 表示指数 4 位全 1（15），尾数 3 位全 1（7），这是 NaN。
    if (f == 0 && a == 127)
        v = NAN;
    //E5M2：指数 5 位全 1（31）。如果尾数非零，则为 NaN；如果尾数为零，则为无穷大。
    else if (f == 1 && e == 31)
        v = m ? NAN : INFINITY;
    // 指数 e 非零且不是特殊值，是正规数。
    // e - bias：无偏指数。
    // + 127：加上 FP32 的偏置，得到 FP32 指数域。
    // << 23：放到 FP32 的指数位置。
    // m << (23 - mb)：把尾数左移到 FP32 尾数域的高位。FP32 会自动补上隐含的 1。
    // frombits：把 32 位整数解释为 float。
    else if (e)
        v = frombits(((e - bias + 127) << 23) | (m << (23 - mb)));
    // 指数 e == 0，是非正规数。
    // 非正规数的实际指数固定为 1 - bias，尾数没有隐含 1，所以值是 m * 2^(1 - bias - mb)。
    // frombits(unsigned(128 - bias - mb) << 23) 构造一个 FP32 数，其指数域为 128 - bias - mb，
    // 实际指数为 (128 - bias - mb) - 127 = 1 - bias - mb，即 2^(1 - bias - mb)。
    // 乘以 float(m) 得到非正规数的值。

    // 因为在非正规数里，有效数字是 0.M，而不是 1.M。
    // 0.M 的值等于：0.M=M/2^(mb)=M*2^(−mb)
    // 所以非正规数的实际值是：(−1)^S*2^(1−bias)*(M/2^(mb))=(−1)^S*M*2^(1−bias−mb)
    else
        v = float(m) * frombits(unsigned(128 - bias - mb) << 23);
    // 根据原始 c 的符号位决定正负，返回 double。
    return c & sign ? -double(v) : double(v);
}
HD inline uint8_t encode_fast(double x, int f, bool stochastic = false, uint64_t seed = 42,
                              uint64_t i = 0) {
    //E4M3：127 是 NaN 的位模式。
    //E5M2：126 是规范 NaN 的位模式。
    //E2M1：没有 NaN，直接返回 0。
    if (std::isnan(x))
        return f == 0 ? 127 : f == 1 ? 126 : 0;
    //确定符号位：如果 x 为负，f=2 时符号位为 8，否则为 128；如果为正，符号位为 0。
    //std::signbit(x) 用来判断一个浮点数的符号位是不是负的，返回 bool：
    unsigned s = std::signbit(x) ? (f == 2 ? 8 : 128) : 0;
    x = fabs(x);
    int top = maxcode(f);
    if (x >= decode_fast(top, f))
        return s | top;
    int lo = 0, hi = top;
    while (hi - lo > 1) {
        int mid = (hi + lo) / 2;
        if (decode_fast(mid, f) <= x)
            lo = mid;
        else
            hi = mid;
    }
    double a = decode_fast(lo, f), b = decode_fast(hi, f);
    // 随机舍入（stochastic = true）
    // uniform(seed, i) 生成 [0, 1) 之间的随机数。
    // 如果随机数小于 (x - a) / (b - a)，就选 hi，否则选 lo。
    // 这样选 hi 的概率正好等于 x 在 [a, b] 中的相对位置，期望值无偏。
    
    // 四舍五入到最近偶数（stochastic = false）
    // x - a > b - x：x 离 b 更近，选 hi。
    // x - a < b - x：x 离 a 更近，选 lo。
    // x - a == b - x：恰好在中点。此时看 lo 的奇偶：
    // 如果 lo 是奇数（lo & 1 == 1），选 hi，因为相邻的 hi = lo + 1 是偶数。
    // 如果 lo 是偶数，保持 lo。
    // 这就是 round-half-to-even，避免统计偏差。
    bool up = stochastic ? uniform(seed, i) < (x - a) / (b - a)
                         : (x - a > b - x || (x - a == b - x && (lo & 1)));
    return s | (up ? hi : lo);
}
} // namespace lp
