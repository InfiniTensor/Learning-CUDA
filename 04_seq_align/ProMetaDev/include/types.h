// types.h - 序列比对的基础类型
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace sa {

// 参考基因组拼接时的填充长度：首尾各留 kRefPadLen 个 'N'，
// 相邻序列之间也用 kRefPadLen 个 'N' 隔开。
// 取值需 > 2*max_band，保证比对窗口不会跨越序列边界、也不会越出数组。
constexpr int64_t kRefPadLen = 64;

// ======================= 参考基因组 =======================
// 多条序列拼接为一个连续碱基数组。为了阻止 k-mer 与比对窗口跨越序列边界，
// 在相邻序列之间插入 (k-1) 个 'N' 作为分隔符（N 不参与索引，且恒为错配）。
struct RefGenome {
    std::vector<std::string> names; // 序列名（不含 '>'）
    std::vector<int64_t> lens;      // 每条序列真实长度
    std::string bases;              // 拼接后的碱基（含分隔符 'N'）
    std::vector<int64_t> seq_begin; // 长度 = names.size()+1，第 i 条序列在 bases 中的起始

    int32_t num_seqs() const {
        return (int32_t)names.size();
    }
    int64_t total_bases() const {
        return (int64_t)bases.size();
    }
    int64_t real_bases() const {
        int64_t s = 0;
        for (int64_t l : lens)
            s += l;
        return s;
    }
};

// ======================= 测序 read =======================
struct Read {
    std::string name;
    std::string bases;
    std::string qual;        // 质量分数（默认不使用）
    int64_t origin_seq = -1; // ground truth（生成器写出时为真值，否则 -1）
    int64_t origin_pos = -1;
};

// ======================= 比对结果 =======================
struct Hit {
    int32_t seq_id = -1; // -1 表示 unknown_origin
    int64_t pos = -1;    // 在所属序列内的 0-based 坐标
    int32_t score = 0;

    bool known() const {
        return seq_id >= 0;
    }
};

// ======================= 算法配置 =======================
struct Config {
    int k = 15;                   // 种子长度（1..16）
    int band = 8;                 // 带状 DP 半宽（允许的最大净空位数）
    int match = 2;                // 匹配得分
    int mismatch = -1;            // 错配得分
    int gap = -1;                 // 空位罚分（线性，取负值）
    int seed_step = 8;            // read 上相邻种子的步长
    int max_candidates = 64;      // 每条 read 最多验证的候选位点数
    double min_score_ratio = 0.7; // 判定阈值：score >= ratio * match * read_len 才算已比对
    int index_bits = 0;           // k-mer 索引哈希桶位数；0 = 按参考规模自动选取
};

} // namespace sa
