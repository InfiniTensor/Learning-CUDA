#include "gvs/io.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace gvs {
namespace {

constexpr char kDatabaseMagic[8] = {'G','V','S','D','B','0','1','\0'};
constexpr char kQueryMagic[8] = {'G','V','S','Q','R','0','1','\0'};
constexpr char kResultMagic[8] = {'G','V','S','R','S','0','1','\0'};

template <typename T>
void write_scalar(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("failed to write binary file");
}

template <typename T>
T read_scalar(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("truncated binary file");
    return value;
}

std::string trim(std::string value) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
        value = value.substr(1, value.size() - 2);
    return value;
}

}  // namespace

void write_vector_file(const std::string& path, const VectorFile& data,
                       bool include_metric) {
    if (data.values.size() != data.count * data.dim)
        throw std::invalid_argument("vector file shape mismatch");
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open output file: " + path);
    out.write(include_metric ? kDatabaseMagic : kQueryMagic, 8);
    write_scalar(out, static_cast<std::uint64_t>(data.count));
    write_scalar(out, static_cast<std::uint32_t>(data.dim));
    write_scalar(out, static_cast<std::uint32_t>(1));  // dtype: 1 = fp32
    if (include_metric)
        write_scalar(out, static_cast<std::uint32_t>(data.metric == Metric::L2 ? 0 : 1));
    out.write(reinterpret_cast<const char*>(data.values.data()),
              static_cast<std::streamsize>(data.values.size() * sizeof(float)));
    if (!out) throw std::runtime_error("failed to write vectors");
}

VectorFile read_vector_file(const std::string& path, bool expect_metric) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open input file: " + path);
    char magic[8]{};
    in.read(magic, 8);
    const char* expected = expect_metric ? kDatabaseMagic : kQueryMagic;
    if (!in || std::memcmp(magic, expected, 8) != 0)
        throw std::runtime_error("invalid vector file magic/version: " + path);
    VectorFile result;
    result.count = static_cast<std::size_t>(read_scalar<std::uint64_t>(in));
    result.dim = static_cast<std::size_t>(read_scalar<std::uint32_t>(in));
    const auto dtype = read_scalar<std::uint32_t>(in);
    if (dtype != 1) throw std::runtime_error("this baseline supports fp32 files only");
    if (expect_metric) {
        const auto metric = read_scalar<std::uint32_t>(in);
        if (metric > 1) throw std::runtime_error("invalid metric code");
        result.metric = metric == 0 ? Metric::L2 : Metric::InnerProduct;
    }
    if (result.count == 0 || result.dim == 0 ||
        result.count > static_cast<std::size_t>(-1) / result.dim)
        throw std::runtime_error("invalid or overflowing vector shape");
    result.values.resize(result.count * result.dim);
    in.read(reinterpret_cast<char*>(result.values.data()),
            static_cast<std::streamsize>(result.values.size() * sizeof(float)));
    if (!in) throw std::runtime_error("truncated vector payload");
    return result;
}

void write_search_result(const std::string& path, const SearchResult& result) {
    if (result.ids.size() != result.queries * result.k ||
        result.scores.size() != result.ids.size())
        throw std::invalid_argument("result shape mismatch");
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open result file: " + path);
    out.write(kResultMagic, 8);
    write_scalar(out, static_cast<std::uint64_t>(result.queries));
    write_scalar(out, static_cast<std::uint32_t>(result.k));
    for (std::size_t i = 0; i < result.ids.size(); ++i) {
        write_scalar(out, static_cast<std::int64_t>(result.ids[i]));
        write_scalar(out, result.scores[i]);
    }
}

Parameters read_parameters(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open parameter file: " + path);
    Parameters p;
    std::string line;
    while (std::getline(in, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        const auto equal = line.find('=');
        if (equal == std::string::npos) continue;
        const std::string key = trim(line.substr(0, equal));
        const std::string value = trim(line.substr(equal + 1));
        if (key == "top_k") p.top_k = std::stoull(value);
        else if (key == "search_mode") p.search_mode = value;
        else if (key == "batch_size") p.batch_size = std::stoull(value);
        else if (key == "nlist") p.nlist = std::stoull(value);
        else if (key == "nprobe") p.nprobe = std::stoull(value);
    }
    if (p.top_k == 0 || p.batch_size == 0 || p.nlist == 0 || p.nprobe == 0)
        throw std::runtime_error("all numeric parameters must be positive");
    if (p.search_mode != "exact" && p.search_mode != "ivf_flat")
        throw std::runtime_error("search_mode must be exact or ivf_flat");
    return p;
}

}  // namespace gvs

