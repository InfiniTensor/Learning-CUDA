// dna.h - DNA 碱基编码（host / device 共用）
#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define SA_HD __host__ __device__
#else
#define SA_HD
#endif

namespace sa {

// A=0 C=1 G=2 T=3；其它字符（含 'N'）返回 4，表示"非法碱基"，
// 含非法碱基的 k-mer 不进入索引，比对时恒判为错配。
SA_HD inline int base_code(char c) {
    switch (c) {
    case 'A':
    case 'a':
        return 0;
    case 'C':
    case 'c':
        return 1;
    case 'G':
    case 'g':
        return 2;
    case 'T':
    case 't':
        return 3;
    default:
        return 4;
    }
}

// 返回匹配得分：两侧都是合法碱基且相同 → match；否则 mismatch
SA_HD inline int subst_score(int a, int b, int match, int mismatch) {
    return (a < 4 && a == b) ? match : mismatch;
}

// k-mer 哈希：把 code（2k 位）打散到 [0, 2^bits)
SA_HD inline uint32_t kmer_hash(uint64_t code, uint32_t mask) {
    // Murmur 风格的 finalizer，保证相邻 k-mer 落入不同桶
    uint64_t x = code;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)(x & mask);
}

} // namespace sa
