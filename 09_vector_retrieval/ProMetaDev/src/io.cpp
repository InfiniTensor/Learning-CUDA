// io.cpp - 配置/向量集/结果/索引 的读写实现
#include "io.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>

namespace vs {

// ======================= 枚举名称 =======================
const char* metric_name(Metric m) {
    switch (m) {
    case Metric::L2:
        return "l2";
    case Metric::INNER_PRODUCT:
        return "inner_product";
    default:
        return "cosine";
    }
}
const char* mode_name(SearchMode m) {
    switch (m) {
    case SearchMode::EXACT:
        return "exact";
    case SearchMode::IVF_FLAT:
        return "ivf_flat";
    default:
        return "ivf_pq";
    }
}
const char* dtype_name(Dtype d) {
    return d == Dtype::FP32 ? "fp32" : "fp16";
}

// ======================= 主机端 fp16 软件转换 =======================
static uint16_t f32_to_f16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xFFu) - 127;
    uint32_t mant = u & 0x7FFFFFu;
    if (exp == 128)
        return (uint16_t)(mant ? (sign | 0x7E00u) : (sign | 0x7C00u));
    int32_t he = exp + 15;
    if (he >= 31)
        return (uint16_t)(sign | 0x7C00u);
    if (he <= 0) {
        if (he < -10)
            return (uint16_t)sign;
        mant |= 0x800000u;
        int shift = 14 - he;
        uint32_t hm = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (hm & 1u)))
            hm++;
        return (uint16_t)(sign | hm);
    }
    uint32_t hm = mant >> 13;
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) {
        hm++;
        if (hm == 0x400u) {
            hm = 0;
            he++;
            if (he >= 31)
                return (uint16_t)(sign | 0x7C00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)he << 10) | hm);
}
static float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            while (!(mant & 0x400u)) {
                mant <<= 1;
                exp--;
            }
            exp += 1;
            mant &= 0x3FFu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// ======================= 字符串工具 =======================
static std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a]))
        a++;
    while (b > a && std::isspace((unsigned char)s[b - 1]))
        b--;
    return s.substr(a, b - a);
}
static std::string strip_quotes(std::string s) {
    if (s.size() >= 2 &&
        ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\'')))
        return s.substr(1, s.size() - 2);
    return s;
}
static std::string lower(std::string s) {
    for (char& c : s)
        c = (char)std::tolower((unsigned char)c);
    return s;
}

// ======================= 配置 =======================
Config load_config(const std::string& path) {
    Config cfg;
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "[io] 无法打开配置文件: %s\n", path.c_str());
        return cfg;
    }
    std::string line;
    while (std::getline(f, line)) {
        size_t hash = line.find('#');
        if (hash != std::string::npos)
            line = line.substr(0, hash);
        line = trim(line);
        if (line.empty())
            continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos)
            continue;
        std::string key = lower(trim(line.substr(0, eq)));
        std::string val = strip_quotes(trim(line.substr(eq + 1)));
        if (key == "top_k")
            cfg.top_k = std::atoi(val.c_str());
        else if (key == "batch_size")
            cfg.batch_size = std::atoi(val.c_str());
        else if (key == "nlist")
            cfg.nlist = std::atoi(val.c_str());
        else if (key == "nprobe")
            cfg.nprobe = std::atoi(val.c_str());
        else if (key == "pq_m")
            cfg.pq_m = std::atoi(val.c_str());
        else if (key == "kmeans_iters")
            cfg.kmeans_iters = std::atoi(val.c_str());
        else if (key == "exact_block")
            cfg.exact_block = std::atoi(val.c_str());
        else if (key == "iters_per_thread")
            cfg.iters_per_thread = std::atoi(val.c_str());
        else if (key == "seed")
            cfg.seed = (uint32_t)std::strtoul(val.c_str(), nullptr, 10);
        else if (key == "no_cpu")
            cfg.no_cpu = (lower(val) == "true" || val == "1");
        else if (key == "search_mode") {
            std::string v = lower(val);
            cfg.mode = (v == "ivf_flat") ? SearchMode::IVF_FLAT
                       : (v == "ivf_pq") ? SearchMode::IVF_PQ
                                         : SearchMode::EXACT;
        }
    }
    if (cfg.top_k < 1)
        cfg.top_k = 1;
    if (cfg.batch_size < 1)
        cfg.batch_size = 1;
    return cfg;
}

// ======================= 向量集 =======================
VectorSet load_vectors(const std::string& path) {
    VectorSet s;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[io] 无法打开向量文件: %s\n", path.c_str());
        return s;
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t dpos = content.find("[data]");
    if (dpos == std::string::npos) {
        std::fprintf(stderr, "[io] 向量文件缺少 [data] 段: %s\n", path.c_str());
        return s;
    }
    std::istringstream hs(content.substr(0, dpos));
    std::string line;
    while (std::getline(hs, line)) {
        line = trim(line);
        size_t colon = line.find(':');
        if (colon == std::string::npos)
            continue;
        std::string key = lower(trim(line.substr(0, colon)));
        std::string val = trim(line.substr(colon + 1));
        if (key == "num_vectors" || key == "num_queries")
            s.n = std::strtoll(val.c_str(), nullptr, 10);
        else if (key == "dim")
            s.dim = (int32_t)std::atoi(val.c_str());
        else if (key == "dtype")
            s.dtype = (lower(val) == "fp16") ? Dtype::FP16 : Dtype::FP32;
        else if (key == "metric") {
            std::string v = lower(val);
            s.metric = (v == "inner_product") ? Metric::INNER_PRODUCT
                       : (v == "cosine")      ? Metric::COSINE
                                              : Metric::L2;
        }
    }
    size_t nl = content.find('\n', dpos);
    if (nl == std::string::npos) {
        std::fprintf(stderr, "[io] [data] 段后无数据\n");
        return s;
    }
    size_t off = nl + 1;
    const char* p = content.data() + off;
    size_t avail = content.size() - off;
    int64_t total = s.n * s.dim;
    s.data.resize((size_t)std::max<int64_t>(total, 0));
    if (s.dtype == Dtype::FP16) {
        if ((int64_t)avail < total * 2) {
            std::fprintf(stderr, "[io] fp16 数据不足: 需要 %lld 字节, 实际 %zu\n",
                         (long long)(total * 2), avail);
            s.data.clear();
            return s;
        }
        const uint16_t* hp = reinterpret_cast<const uint16_t*>(p);
        for (int64_t i = 0; i < total; i++)
            s.data[(size_t)i] = f16_bits_to_f32(hp[i]);
    } else {
        if ((int64_t)avail < total * 4) {
            std::fprintf(stderr, "[io] fp32 数据不足: 需要 %lld 字节, 实际 %zu\n",
                         (long long)(total * 4), avail);
            s.data.clear();
            return s;
        }
        std::memcpy(s.data.data(), p, (size_t)(total * 4));
    }
    return s;
}

void save_vectors(const std::string& path, const VectorSet& vs) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[io] 无法写出向量文件: %s\n", path.c_str());
        return;
    }
    f << "[header]\n";
    f << "num_vectors: " << vs.n << "\n";
    f << "dim: " << vs.dim << "\n";
    f << "dtype: " << dtype_name(vs.dtype) << "\n";
    f << "metric: " << metric_name(vs.metric) << "\n\n";
    f << "[data]\n";
    int64_t total = vs.n * vs.dim;
    if (vs.dtype == Dtype::FP16) {
        std::vector<uint16_t> tmp((size_t)total);
        for (int64_t i = 0; i < total; i++)
            tmp[(size_t)i] = f32_to_f16_bits(vs.data[(size_t)i]);
        f.write(reinterpret_cast<const char*>(tmp.data()), (std::streamsize)(total * 2));
    } else {
        f.write(reinterpret_cast<const char*>(vs.data.data()), (std::streamsize)(total * 4));
    }
}

// ======================= 合成数据 =======================
// 优化：对高维球面/簇结构使用低差异的随机方向和半径，使数据分布更接近真实嵌入
VectorSet gen_vectors(int64_t n, int32_t dim, const std::string& kind, uint32_t seed, int nlist) {
    VectorSet s;
    s.n = n;
    s.dim = dim;
    s.dtype = Dtype::FP32;
    s.metric = Metric::L2;
    s.data.resize((size_t)(n * dim));
    std::mt19937 rng(seed);
    std::string k = lower(kind);

    if (k == "clustered" && nlist > 0) {
        // nlist 个高斯簇：中心 ~ N(0,1)，样本 = 中心 + 0.1*N(0,1)
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::normal_distribution<float> pert(0.0f, 0.1f);
        std::vector<float> centers((size_t)((int64_t)nlist * dim));
        for (auto& c : centers)
            c = g(rng);
        std::uniform_int_distribution<int> pick(0, nlist - 1);
        for (int64_t i = 0; i < n; i++) {
            int c = pick(rng);
            for (int32_t d = 0; d < dim; d++)
                s.data[(size_t)(i * dim + d)] = centers[(size_t)((int64_t)c * dim + d)] + pert(rng);
        }
        return s;
    }

    for (int64_t i = 0; i < n; i++) {
        if (k == "normal") {
            std::normal_distribution<float> g(0.0f, 1.0f);
            for (int32_t d = 0; d < dim; d++)
                s.data[(size_t)(i * dim + d)] = g(rng);
        } else {
            std::uniform_real_distribution<float> u(-1.0f, 1.0f);
            for (int32_t d = 0; d < dim; d++)
                s.data[(size_t)(i * dim + d)] = u(rng);
        }
    }
    return s;
}

VectorSet gen_queries(int64_t nq, const VectorSet& base, const std::string& kind, uint32_t seed) {
    VectorSet q;
    q.n = nq;
    q.dim = base.dim;
    q.dtype = Dtype::FP32;
    q.metric = base.metric;
    q.data.resize((size_t)(nq * base.dim));
    std::mt19937 rng(seed);
    std::string k = lower(kind);

    if (k == "clustered" && base.n > 0) {
        // 查询 = 随机取一条库向量 + 小扰动（更贴近真实查询分布）
        std::normal_distribution<float> pert(0.0f, 0.05f);
        std::uniform_int_distribution<int64_t> pick(0, base.n - 1);
        for (int64_t i = 0; i < nq; i++) {
            int64_t src = pick(rng);
            for (int32_t d = 0; d < base.dim; d++)
                q.data[(size_t)(i * base.dim + d)] =
                    base.data[(size_t)(src * base.dim + d)] + pert(rng);
        }
        return q;
    }
    for (int64_t i = 0; i < nq; i++) {
        if (k == "normal") {
            std::normal_distribution<float> g(0.0f, 1.0f);
            for (int32_t d = 0; d < base.dim; d++)
                q.data[(size_t)(i * base.dim + d)] = g(rng);
        } else {
            std::uniform_real_distribution<float> u(-1.0f, 1.0f);
            for (int32_t d = 0; d < base.dim; d++)
                q.data[(size_t)(i * base.dim + d)] = u(rng);
        }
    }
    return q;
}

// ======================= 结果文件 =======================
void save_result(const std::string& path, const SearchResult& r) {
    std::ofstream f(path);
    if (!f) {
        std::fprintf(stderr, "[io] 无法写出结果文件: %s\n", path.c_str());
        return;
    }
    f << "# query_id rank vector_id distance\n";
    for (int64_t i = 0; i < r.nq; i++) {
        for (int j = 0; j < r.k; j++) {
            size_t idx = (size_t)(i * r.k + j);
            f << i << ' ' << j << ' ' << r.ids[idx] << ' ' << r.dists[idx] << '\n';
        }
    }
}

SearchResult load_result(const std::string& path, int64_t nq, int k) {
    SearchResult r;
    r.nq = nq;
    r.k = k;
    r.ids.assign((size_t)(nq * k), -1);
    r.dists.assign((size_t)(nq * k), 0.0f);
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "[io] 无法打开结果文件: %s\n", path.c_str());
        return r;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream ss(line);
        long long qi, rank, id;
        float dist;
        if (!(ss >> qi >> rank >> id >> dist))
            continue;
        if (qi < 0 || qi >= nq || rank < 0 || rank >= k)
            continue;
        size_t idx = (size_t)(qi * k + rank);
        r.ids[idx] = (int32_t)id;
        r.dists[idx] = dist;
    }
    return r;
}

// ======================= 索引文件 =======================
#pragma pack(push, 1)
struct IndexHeader {
    char magic[4];
    int32_t nlist;
    int32_t dim;
    int32_t metric;
    int32_t pq_m;
    int64_t n;
    int64_t centroids_count;
    int64_t list_start_count;
    int64_t list_ids_count;
    int64_t pq_codebook_count;
    int64_t pq_codes_count;
};
#pragma pack(pop)

void save_index(const std::string& path, const IvfIndex& idx) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[io] 无法写出索引文件: %s\n", path.c_str());
        return;
    }
    IndexHeader h{};
    std::memcpy(h.magic, "VSI1", 4);
    h.nlist = idx.nlist;
    h.dim = idx.dim;
    h.metric = (int32_t)idx.metric;
    h.pq_m = idx.pq_m;
    h.n = (int64_t)idx.list_ids.size();
    h.centroids_count = (int64_t)idx.centroids.size();
    h.list_start_count = (int64_t)idx.list_start.size();
    h.list_ids_count = (int64_t)idx.list_ids.size();
    h.pq_codebook_count = (int64_t)idx.pq_codebook.size();
    h.pq_codes_count = (int64_t)idx.pq_codes.size();
    f.write(reinterpret_cast<const char*>(&h), sizeof(IndexHeader));
    if (h.centroids_count > 0)
        f.write(reinterpret_cast<const char*>(idx.centroids.data()),
                (std::streamsize)(h.centroids_count * 4));
    if (h.list_start_count > 0)
        f.write(reinterpret_cast<const char*>(idx.list_start.data()),
                (std::streamsize)(h.list_start_count * 4));
    if (h.list_ids_count > 0)
        f.write(reinterpret_cast<const char*>(idx.list_ids.data()),
                (std::streamsize)(h.list_ids_count * 4));
    if (h.pq_codebook_count > 0)
        f.write(reinterpret_cast<const char*>(idx.pq_codebook.data()),
                (std::streamsize)(h.pq_codebook_count * 4));
    if (h.pq_codes_count > 0)
        f.write(reinterpret_cast<const char*>(idx.pq_codes.data()),
                (std::streamsize)h.pq_codes_count);
}

bool load_index(const std::string& path, IvfIndex& idx) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[io] 无法打开索引文件: %s\n", path.c_str());
        return false;
    }
    IndexHeader h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(IndexHeader));
    if (std::memcmp(h.magic, "VSI1", 4) != 0) {
        std::fprintf(stderr, "[io] 索引文件 magic 不匹配\n");
        return false;
    }
    idx.nlist = h.nlist;
    idx.dim = h.dim;
    idx.metric = (Metric)h.metric;
    idx.pq_m = h.pq_m;
    idx.centroids.resize((size_t)h.centroids_count);
    idx.list_start.resize((size_t)h.list_start_count);
    idx.list_ids.resize((size_t)h.list_ids_count);
    idx.pq_codebook.resize((size_t)h.pq_codebook_count);
    idx.pq_codes.resize((size_t)h.pq_codes_count);
    if (h.centroids_count > 0)
        f.read(reinterpret_cast<char*>(idx.centroids.data()),
               (std::streamsize)(h.centroids_count * 4));
    if (h.list_start_count > 0)
        f.read(reinterpret_cast<char*>(idx.list_start.data()),
               (std::streamsize)(h.list_start_count * 4));
    if (h.list_ids_count > 0)
        f.read(reinterpret_cast<char*>(idx.list_ids.data()),
               (std::streamsize)(h.list_ids_count * 4));
    if (h.pq_codebook_count > 0)
        f.read(reinterpret_cast<char*>(idx.pq_codebook.data()),
               (std::streamsize)(h.pq_codebook_count * 4));
    if (h.pq_codes_count > 0)
        f.read(reinterpret_cast<char*>(idx.pq_codes.data()), (std::streamsize)h.pq_codes_count);
    return true;
}

} // namespace vs
