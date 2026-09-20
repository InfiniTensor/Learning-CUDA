// io.h - FASTA / FASTQ / 结果文件读写
#pragma once
#include "types.h"
#include <string>
#include <vector>

namespace sa {

// 由序列名与碱基序列构造拼接参考（applied 填充规则见 types.h 的 kRefPadLen）
RefGenome make_reference(const std::vector<std::string>& names,
                         const std::vector<std::string>& seqs, int k);

// 读取 FASTA 参考基因组，并按 k 插入填充/分隔符构造拼接序列
RefGenome load_reference(const std::string& path, int k);

// 读取 FASTQ reads（第 3、4 行（质量）默认忽略）
std::vector<Read> load_reads(const std::string& path);

// 写出比对结果：<read名> <参考序列名> <起始位置> <得分>；未比对则 <read名> unknown_origin
void write_hits(const std::string& path, const RefGenome& ref, const std::vector<Read>& reads,
                const std::vector<Hit>& hits);

// 读取真值文件（生成器写出，格式：<seq_id> <pos> <score>；-1 -1 0 表示无来源）
// 用于大规模测试时统计召回率
std::vector<Hit> load_truth(const std::string& path);

} // namespace sa
