// io.cpp - FASTA / FASTQ / 结果文件读写实现
#include "io.h"
#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace sa {

namespace {

// 去掉行尾 \r \n 与空白
void chomp(std::string& s) {
    while (!s.empty() &&
           (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
}

// 取名字中第一个空白之前的部分
std::string first_token(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && s[i] != ' ' && s[i] != '\t')
        ++i;
    return s.substr(0, i);
}

} // namespace

RefGenome make_reference(const std::vector<std::string>& names,
                         const std::vector<std::string>& seqs, int k) {
    if (names.size() != seqs.size() || names.empty())
        throw std::runtime_error("序列名与序列数量不一致");

    RefGenome ref;
    ref.names = names;
    ref.lens.resize(seqs.size());
    for (size_t i = 0; i < seqs.size(); ++i)
        ref.lens[i] = (int64_t)seqs[i].size();

    // 首尾各 kRefPadLen 个 'N'；序列之间用 max(k-1, kRefPadLen) 个 'N' 隔开。
    // 目的：① k-mer 不会跨越序列边界；② 比对窗口不会越出数组、也不会跨越边界。
    const int64_t sep = std::max<int64_t>((k > 1) ? (k - 1) : 0, kRefPadLen);
    ref.seq_begin.resize(seqs.size() + 1);
    int64_t pos = kRefPadLen;
    for (size_t i = 0; i < seqs.size(); ++i) {
        if (i > 0)
            pos += sep;
        ref.seq_begin[i] = pos;
        pos += ref.lens[i];
    }
    ref.seq_begin[seqs.size()] = pos;
    pos += kRefPadLen;

    ref.bases.assign((size_t)pos, 'N');
    for (size_t i = 0; i < seqs.size(); ++i) {
        ref.bases.replace((size_t)ref.seq_begin[i], (size_t)ref.lens[i], seqs[i]);
    }
    return ref;
}

RefGenome load_reference(const std::string& path, int k) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("无法打开参考基因组文件: " + path);

    // 一次性读入整个文件再在内存里解析：避免逐行 getline 的重复拷贝与跨界开销
    f.seekg(0, std::ios::end);
    const std::streamoff sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if (sz <= 0)
        throw std::runtime_error("参考基因组文件为空: " + path);

    std::string buf;
    buf.resize((size_t)sz);
    f.read(&buf[0], sz);
    buf.resize((size_t)f.gcount());

    std::vector<std::string> names, seqs;
    std::string cur;
    const size_t n = buf.size();
    size_t i = 0;
    bool in_seq = false;

    while (i < n) {
        size_t eol = buf.find('\n', i);
        if (eol == std::string::npos)
            eol = n;
        size_t len = eol - i;
        if (len > 0 && buf[i + len - 1] == '\r')
            --len; // 兼容 CRLF
        if (len > 0 && buf[i] == '>') {
            if (in_seq) {
                seqs.push_back(std::move(cur));
                cur.clear();
            }
            names.push_back(first_token(buf.substr(i + 1, len - 1)));
            in_seq = true;
        } else if (in_seq && len > 0) {
            cur.append(buf, i, len);
        }
        i = eol + 1;
    }
    if (in_seq)
        seqs.push_back(std::move(cur));
    if (seqs.empty())
        throw std::runtime_error("参考基因组为空: " + path);

    return make_reference(names, seqs, k);
}

std::vector<Read> load_reads(const std::string& path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("无法打开 reads 文件: " + path);

    std::vector<Read> reads;
    std::string l1, l2, l3, l4;
    while (std::getline(f, l1)) {
        chomp(l1);
        if (l1.empty())
            continue;
        if (l1[0] != '@')
            throw std::runtime_error("FASTQ 格式错误（首行应以 @ 开头）: " + l1);
        if (!std::getline(f, l2))
            break;
        if (!std::getline(f, l3))
            break;
        if (!std::getline(f, l4))
            break;
        chomp(l2);
        chomp(l3);
        chomp(l4);

        Read r;
        r.name = first_token(l1.substr(1));
        r.bases = l2;
        r.qual = l4;
        reads.push_back(std::move(r));
    }
    if (reads.empty())
        throw std::runtime_error("reads 文件为空: " + path);
    return reads;
}

void write_hits(const std::string& path, const RefGenome& ref, const std::vector<Read>& reads,
                const std::vector<Hit>& hits) {
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("无法写入结果文件: " + path);
    if (reads.size() != hits.size())
        throw std::runtime_error("reads 与 hits 数量不一致");

    for (size_t i = 0; i < reads.size(); ++i) {
        const Hit& h = hits[i];
        if (!h.known()) {
            f << reads[i].name << " unknown_origin\n";
        } else {
            f << reads[i].name << " " << ref.names[(size_t)h.seq_id] << " " << h.pos << " "
              << h.score << "\n";
        }
    }
}

std::vector<Hit> load_truth(const std::string& path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("无法打开真值文件: " + path);
    std::vector<Hit> v;
    int32_t seq_id, score;
    int64_t pos;
    while (f >> seq_id >> pos >> score) {
        Hit h;
        h.seq_id = seq_id;
        h.pos = pos;
        h.score = score;
        v.push_back(h);
    }
    return v;
}

} // namespace sa
