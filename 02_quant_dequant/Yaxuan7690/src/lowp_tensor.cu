// =============================================================================
//  lowp_tensor.cu -- 张量文件 / 低精度权重文件 的读写实现
//
//  优化 21(宿主侧 IO): 整块文件读写不再走 iostream.
//  实测(256MiB, 本机 D 盘, 已进页缓存): std::ifstream 162ms(1.66GB/s) /
//  std::ofstream 120ms(1.17GB/s), 同一块盘上 CRT 的 fread 39ms(6.88GB/s) /
//  fwrite 81ms(1.73GB/s) -- 读相差 4.2x. 原实现里光是「把 256MiB 输入读进来」
//  这一项就是整个 GPU 部分(1.4~1.9ms)的 100 倍, 属于端到端里最大的一块浪费.
//  所以这里加一层平台相关的整块读写:
//      Windows : _wfopen(UTF-8 -> UTF-16 路径) + fread / fwrite
//      其它    : std::ifstream / std::ofstream(与原先一致, 由 Makefile 平台使用)
//
//  同时把「整文件读进临时 buffer, 再 assign 进目标容器」这种两遍拷贝去掉:
//  张量文件先读文件头算出数据偏移, 再直接 seek 过去读进 HostTensor::data;
//  低精度权重文件同理, 头 + packed + scales 三段各自直接读进目标容器,
//  写侧也改成头/packed/scales 三段顺序写出, 不再先拼一个大 buffer.
// =============================================================================
#include "lpq/lowp_tensor.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>

#include "lpq/common.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace lpq {
namespace {

// -----------------------------------------------------------------------------
// 平台相关的整块文件读写
// -----------------------------------------------------------------------------
#ifdef _WIN32
std::wstring utf8_to_wide(const std::string& s) {
    std::wstring w;
    if (s.empty())
        return w;
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    if (n <= 0)
        return w;
    w.resize((size_t)n);
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), &w[0], n);
    return w;
}
#endif

// 只读文件流: seek + 整块读
class FileReader {
public:
    explicit FileReader(const std::string& path) {
#ifdef _WIN32
        f_ = _wfopen(utf8_to_wide(path).c_str(), L"rb");
#else
        ifs_.open(path, std::ios::binary);
#endif
    }

    ~FileReader() { close(); }

    FileReader(const FileReader&) = delete;
    FileReader& operator=(const FileReader&) = delete;

    bool ok() const {
#ifdef _WIN32
        return f_ != nullptr;
#else
        return ifs_.is_open() && ifs_.good();
#endif
    }

    // 文件总字节数;失败返回 0
    uint64_t size() {
        if (!ok())
            return 0;
#ifdef _WIN32
        if (_fseeki64(f_, 0, SEEK_END) != 0)
            return 0;
        const __int64 n = _ftelli64(f_);
        if (_fseeki64(f_, 0, SEEK_SET) != 0)
            return 0;
        return n > 0 ? (uint64_t)n : 0;
#else
        ifs_.seekg(0, std::ios::end);
        const std::streamoff n = ifs_.tellg();
        ifs_.seekg(0, std::ios::beg);
        return n > 0 ? (uint64_t)n : 0;
#endif
    }

    bool seek(uint64_t off) {
        if (!ok())
            return false;
#ifdef _WIN32
        return _fseeki64(f_, (__int64)off, SEEK_SET) == 0;
#else
        ifs_.clear();
        ifs_.seekg((std::streamoff)off, std::ios::beg);
        return (bool)ifs_;
#endif
    }

    size_t read(void* dst, size_t bytes) {
        if (!ok() || bytes == 0)
            return 0;
#ifdef _WIN32
        return fread(dst, 1, bytes, f_);
#else
        ifs_.read(reinterpret_cast<char*>(dst), (std::streamsize)bytes);
        return (size_t)ifs_.gcount();
#endif
    }

    void close() {
#ifdef _WIN32
        if (f_ != nullptr) {
            fclose(f_);
            f_ = nullptr;
        }
#else
        if (ifs_.is_open())
            ifs_.close();
#endif
    }

private:
#ifdef _WIN32
    FILE* f_ = nullptr;
#else
    std::ifstream ifs_;
#endif
};

// 顺序写文件流(打开即截断)
class FileWriter {
public:
    explicit FileWriter(const std::string& path) {
#ifdef _WIN32
        f_ = _wfopen(utf8_to_wide(path).c_str(), L"wb");
#else
        ofs_.open(path, std::ios::binary | std::ios::trunc);
#endif
    }

    ~FileWriter() { close(); }

    FileWriter(const FileWriter&) = delete;
    FileWriter& operator=(const FileWriter&) = delete;

    bool ok() const {
#ifdef _WIN32
        return f_ != nullptr;
#else
        return ofs_.is_open() && ofs_.good();
#endif
    }

    bool write(const void* data, size_t bytes) {
        if (!ok())
            return false;
        if (bytes == 0)
            return true;
#ifdef _WIN32
        return fwrite(data, 1, bytes, f_) == bytes;
#else
        ofs_.write(reinterpret_cast<const char*>(data), (std::streamsize)bytes);
        return (bool)ofs_;
#endif
    }

    void close() {
#ifdef _WIN32
        if (f_ != nullptr) {
            fclose(f_);
            f_ = nullptr;
        }
#else
        if (ofs_.is_open())
            ofs_.close();
#endif
    }

private:
#ifdef _WIN32
    FILE* f_ = nullptr;
#else
    std::ofstream ofs_;
#endif
};

void write_all_bytes(const std::string& path, const void* data, size_t bytes) {
    FileWriter w(path);
    if (!w.ok())
        throw Error("cannot write file: " + path, __FILE__, __LINE__);
    if (!w.write(data, bytes))
        throw Error("write to file failed: " + path, __FILE__, __LINE__);
}

template <class T>
T load_le(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

}  // namespace

HostTensor read_tensor_file(const std::string& path) {
    FileReader r(path);
    if (!r.ok())
        throw Error("cannot read tensor file: " + path, __FILE__, __LINE__);

    const uint64_t fsize = r.size();
    if (fsize < 17)
        throw Error("tensor file is too short (no header): " + path, __FILE__, __LINE__);

    // 只读文件头(shape / dtype)就够定位数据了, 不必先读整个文件
    uint8_t head[17 + 255] = {0};
    const size_t head_n = (size_t)(fsize < sizeof(head) ? fsize : sizeof(head));
    if (r.read(head, head_n) != head_n)
        throw Error("cannot read tensor file: " + path, __FILE__, __LINE__);

    HostTensor t;
    t.rows = load_le<int64_t>(head);
    t.cols = load_le<int64_t>(head + 8);
    const uint8_t dtype_len = head[16];
    if (17u + (unsigned)dtype_len > head_n)
        throw Error("dtype length in tensor header is out of range: " + path, __FILE__, __LINE__);
    std::string dtype(reinterpret_cast<const char*>(head + 17), dtype_len);
    t.dtype = input_type_from_string(dtype);

    const size_t elem = t.elem_size();
    const size_t need = (size_t)t.rows * (size_t)t.cols * elem;
    // 兼容两种数据起始偏移:紧跟 dtype 串 / 8 字节对齐
    const size_t off_plain = 17u + (unsigned)dtype_len;
    const size_t off_aligned = (off_plain + 7u) & ~size_t(7);
    size_t off;
    if (fsize >= off_plain + need) {
        off = off_plain;
    } else if (fsize >= off_aligned + need) {
        off = off_aligned;
    } else {
        throw Error("tensor data is shorter than num_rows * num_cols: " + path, __FILE__, __LINE__);
    }

    // 直接读进目标 buffer(只有这一遍)
    t.data.resize(need);
    if (need > 0 && (!r.seek(off) || r.read(t.data.data(), need) != need))
        throw Error("cannot read tensor file: " + path, __FILE__, __LINE__);
    return t;
}

void write_tensor_file(const std::string& path, const HostTensor& t) {
    const std::string dtype = to_string(t.dtype);
    uint8_t head[17 + 255];
    std::memcpy(head, &t.rows, 8);
    std::memcpy(head + 8, &t.cols, 8);
    head[16] = (uint8_t)dtype.size();
    std::memcpy(head + 17, dtype.data(), dtype.size());

    FileWriter w(path);
    if (!w.ok())
        throw Error("cannot write file: " + path, __FILE__, __LINE__);
    if (!w.write(head, 17 + dtype.size()) || !w.write(t.data.data(), t.data.size()))
        throw Error("write to file failed: " + path, __FILE__, __LINE__);
}

void write_lowp_file(const std::string& path, const LowpFile& f) {
    FileWriter w(path);
    if (!w.ok())
        throw Error("cannot write file: " + path, __FILE__, __LINE__);
    const uint8_t* h = reinterpret_cast<const uint8_t*>(&f.header);
    if (!w.write(h, sizeof(LowpHeader)) || !w.write(f.packed.data(), f.packed.size()) ||
        !w.write(f.scales.data(), f.scales.size())) {
        throw Error("write to file failed: " + path, __FILE__, __LINE__);
    }
}

LowpFile read_lowp_file(const std::string& path) {
    FileReader r(path);
    if (!r.ok())
        throw Error("cannot read lowp weight file: " + path, __FILE__, __LINE__);

    const uint64_t fsize = r.size();
    if (fsize < sizeof(LowpHeader))
        throw Error("lowp weight file is too short: " + path, __FILE__, __LINE__);

    LowpFile f;
    if (r.read(&f.header, sizeof(LowpHeader)) != sizeof(LowpHeader))
        throw Error("cannot read lowp weight file: " + path, __FILE__, __LINE__);
    if (std::memcmp(f.header.magic, "LPQLOWP", 7) != 0)
        throw Error("lowp weight file magic mismatch: " + path, __FILE__, __LINE__);
    if (f.header.packed_bytes < 0 || f.header.scale_bytes < 0)
        throw Error("lowp weight file has a negative payload length: " + path, __FILE__, __LINE__);
    const uint64_t need =
        sizeof(LowpHeader) + (uint64_t)f.header.packed_bytes + (uint64_t)f.header.scale_bytes;
    if (fsize < need)
        throw Error("lowp weight file has an unexpected header size: " + path, __FILE__, __LINE__);

    f.packed.resize((size_t)f.header.packed_bytes);
    f.scales.resize((size_t)f.header.scale_bytes);
    if (!f.packed.empty() && r.read(f.packed.data(), f.packed.size()) != f.packed.size())
        throw Error("cannot read lowp weight file: " + path, __FILE__, __LINE__);
    if (!f.scales.empty() && r.read(f.scales.data(), f.scales.size()) != f.scales.size())
        throw Error("cannot read lowp weight file: " + path, __FILE__, __LINE__);
    return f;
}

void write_raw_file(const std::string& path, const void* data, size_t bytes) {
    write_all_bytes(path, data, bytes);
}

const char* elem_kind_name(uint32_t k) {
    switch (k) {
        case 0:
            return "e4m3";
        case 1:
            return "e5m2";
        case 2:
            return "e2m1";
        default:
            return "unknown";
    }
}

const char* scale_kind_name(uint32_t k) {
    switch (k) {
        case 0:
            return "e8m0";
        case 1:
            return "e4m3";
        default:
            return "unknown";
    }
}

}  // namespace lpq