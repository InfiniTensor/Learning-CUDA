#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <algorithm>

struct QuantState {
    // header
    int num_rows = 0;
    int num_cols = 0;
    int block_size = 0;
    int group_size = 256; // baseline给的是256

    // data (host)
    uint8_t* packed_weights = nullptr;   // 每字节存两个 4-bit 索引
    uint8_t* absmax_q = nullptr;
    __half* absmax2 = nullptr;
    __half code2[256]{};
    float offset = 0.f;

    // runtime param
    std::string compute_type;
    std::string target_gpu;

    int num_elements = 0;
    int num_blocks = 0;
    int num_groups = 0;

    __half* ref_result = nullptr;

    int packed_weights_len_in_bytes = 0;
    int absmax_q_len_in_bytes = 0;
    int absmax2_len_in_bytes = 0;

    void calculate_params() {
        num_elements = num_rows * num_cols;
        // group_size = 256;
        num_blocks = (num_elements + block_size - 1) / block_size;
        num_groups = (num_blocks + group_size - 1) / group_size;

        packed_weights_len_in_bytes = (num_elements + 1) / 2;
        absmax_q_len_in_bytes = num_blocks;
        absmax2_len_in_bytes = 2 * num_groups; // fp16 bytes
    }

    void print() {
        std::cout << "[header]" << std::endl;
        std::cout << "num_rows: " << num_rows << std::endl;
        std::cout << "num_cols: " << num_cols << std::endl;
        std::cout << "blocksize: " << block_size << std::endl;

        std::cout << std::endl;
        std::cout << "[data]" << std::endl;
        std::cout << "packed_weights: " << std::endl;
        int print_cnt = 0;
        for (int i = 0; i < num_elements; i += 2) {
            uint8_t v = packed_weights[i / 2];
            int lower = v & 0xF;
            int upper = v >> 4;
            std::cout << upper << "\t";
            print_cnt ++;
            if (print_cnt == num_cols) {
                std::cout << std::endl;
                print_cnt = 0;
            }
            std::cout << lower << "\t";
            print_cnt ++;
            if (print_cnt == num_cols) {
                std::cout << std::endl;
                print_cnt = 0;
            }
        }
        std::cout << "absmax_q:" << std::endl;
        for (int i = 0; i < num_blocks; ++i) {
            std::cout << (int)absmax_q[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "absmax2:" << std::endl;
        for (int i = 0; i < num_groups; ++i) {
            std::cout << __half2float(absmax2[i]) << " ";
        }
        std::cout << std::endl;

        std::cout << "code2: " << std::endl;
        for (int i = 0; i < 256; ++i) {
            std::cout << __half2float(code2[i]) << " ";
        }
        std::cout << std::endl;

        std::cout << "offset: " << offset << std::endl;

    }
};
// --------- helpers: streaming parse, no full-file scanning ----------

static void expect_text(std::istream& is, const char* s) {
    for (const char* p = s; *p; ++p) {
        char c;
        if (!is.get(c)) {
            throw std::runtime_error(std::string("Unexpected EOF while expecting: ") + s);
        }
        if (c != *p) {
            std::string msg = "Tag mismatch. Expect: ";
            msg += s;
            msg += " (got different byte)";
            throw std::runtime_error(msg);
        }
    }
}

template <typename T>
static T read_pod(std::istream& is) {
    T v{};
    if (!is.read(reinterpret_cast<char*>(&v), sizeof(T))) {
        throw std::runtime_error("Failed to read POD bytes");
    }
    return v; // 假设小端；你写文件也用小端 pack
}

static void read_bytes(std::istream& is, void* dst, size_t n) {
    if (n == 0) return;
    if (!is.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n))) {
        throw std::runtime_error("Failed to read raw bytes");
    }
}

static std::string trim_copy(std::string s) {
    auto not_space = [](unsigned char ch){ return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

static std::string strip_quotes(std::string s) {
    s = trim_copy(std::move(s));
    if (s.size() >= 2) {
        char a = s.front(), b = s.back();
        if ((a == '"' && b == '"') || (a == '\'' && b == '\'')) {
            return s.substr(1, s.size() - 2);
        }
    }
    return s;
}

// input_data: w_nf4.bin
// input_conf: 目前不用（保留接口）
static QuantState parse_quant_state(const std::string& input_data,
        const std::string& input_conf,
        const std::string& ref_result = "") {

    std::ifstream is(input_data, std::ios::binary);
    if (!is) {
        throw std::runtime_error("Failed to open file: " + input_data);
    }

    QuantState st;

    // 你的文件格式（标签文本 + 紧跟二进制）必须严格一致：
    // [header]\n
    // num_rows: <int64>\n
    // num_cols: <int64>\n
    // blocksize: <int32>\n
    //
    // [data]\n
    // packed_weights: <uint8 blob>\n
    // absmax_q: <uint8 blob>\n
    // absmax2: <fp16 blob>\n
    // code2: <fp16[256] blob>\n
    // offset: <float32>\n

    expect_text(is, "[header]\n");

    expect_text(is, "num_rows: ");
    int64_t num_rows64 = read_pod<int64_t>(is);

    expect_text(is, "\nnum_cols: ");
    int64_t num_cols64 = read_pod<int64_t>(is);

    expect_text(is, "\nblocksize: ");
    int32_t blocksize32 = read_pod<int32_t>(is);

    // 注意：QuantState 里用 int，正常矩阵规模不会溢出
    st.num_rows = static_cast<int>(num_rows64);
    st.num_cols = static_cast<int>(num_cols64);
    st.block_size = static_cast<int>(blocksize32);

    st.calculate_params();

    // header 后你写了 "\n\n[data]\n"
    expect_text(is, "\n\n[data]\n");

    expect_text(is, "packed_weights: ");
    st.packed_weights = new uint8_t[st.packed_weights_len_in_bytes];
    read_bytes(is, st.packed_weights, static_cast<size_t>(st.packed_weights_len_in_bytes));

    expect_text(is, "\nabsmax_q: ");
    st.absmax_q = new uint8_t[st.absmax_q_len_in_bytes];
    read_bytes(is, st.absmax_q, static_cast<size_t>(st.absmax_q_len_in_bytes));

    expect_text(is, "\nabsmax2: ");
    st.absmax2 = new __half[st.num_groups];
    read_bytes(is, st.absmax2, static_cast<size_t>(st.absmax2_len_in_bytes));

    expect_text(is, "\ncode2: ");
    read_bytes(is, st.code2, sizeof(__half) * 256);

    expect_text(is, "\noffset: ");
    st.offset = read_pod<float>(is);


    std::ifstream i_conf(input_conf);
    if (!i_conf) {
        throw std::runtime_error("Failed to open conf file: " + input_conf);
    }

    std::string line;

    while (std::getline(i_conf, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back(); // 兼容 CRLF

        // 去掉注释：支持 # 和 //
        auto cut_comment = [&](const std::string& marker) {
            auto pos = line.find(marker);
            if (pos != std::string::npos) line = line.substr(0, pos);
        };
        cut_comment("#");
        cut_comment("//");

        line = trim_copy(line);
        if (line.empty()) continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim_copy(line.substr(0, eq));
        std::string val = trim_copy(line.substr(eq + 1));

        if (key == "blocksize") {
            int bs = std::stoi(val);
            st.block_size = bs;
        } else if (key == "compute_type") {
            st.compute_type = strip_quotes(val);
        } else if (key == "target_gpu") {
            st.target_gpu = strip_quotes(val);
        }
    }

    if (!ref_result.empty()) {
        std::ifstream i_ref_res(ref_result);
        if (!i_ref_res) {
            throw std::runtime_error("Failed to open conf file: " + ref_result);
        }
        st.ref_result = new __half[st.num_elements];
        if (!i_ref_res.read(reinterpret_cast<char*>(st.ref_result), static_cast<std::streamsize>(st.num_elements * 2))) {
            throw std::runtime_error("Failed to read raw bytes");
        }
    }

    return st;
}

