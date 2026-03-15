#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <stdint.h>
#include <algorithm>

// NF4 查找表
const float NF4_TABLE[16] = {
    -1.0f, -0.69487101f, -0.51209301f, -0.37391701f,
    -0.25611401f, -0.14725500f, -0.04162400f, 0.06282201f,
    0.16859101f, 0.28551400f, 0.40619302f, 0.53675699f,
    0.68502200f, 0.87091398f, 1.0f, 0.0f
};

// FP16 转换
float half_to_float_cpu(uint16_t h) {
    union { float f; uint32_t i; } res;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7c00) >> 10;
    uint32_t mant = (h & 0x03ff) << 13;
    if (exp == 0x1f) {
        res.i = sign | 0x7f800000 | mant;
    } else if (exp == 0) {
        if (mant == 0) res.i = sign;
        else {
            exp = 127 - 14;
            while (!(mant & 0x00800000)) { mant <<= 1; exp--; }
            res.i = sign | (exp << 23) | (mant & 0x007fffff);
        }
    } else {
        res.i = sign | ((exp + (127 - 15)) << 23) | mant;
    }
    return res.f;
}

struct WeightHeader {
    int64_t num_rows;
    int64_t num_cols;
    int32_t blocksize;
};

int main() {
    std::ifstream ifs("input.bin", std::ios::binary);
    if (!ifs) return -1;

    // 1. 读取 Header（按字段读取，避免结构体对齐带来的偏移错误）
    WeightHeader header;
    ifs.read(reinterpret_cast<char*>(&header.num_rows), sizeof(header.num_rows));
    ifs.read(reinterpret_cast<char*>(&header.num_cols), sizeof(header.num_cols));
    ifs.read(reinterpret_cast<char*>(&header.blocksize), sizeof(header.blocksize));
    
    int64_t total_elements = header.num_rows * header.num_cols;
    int32_t num_blocks = (total_elements + header.blocksize - 1) / header.blocksize;
    int32_t group_size = 256; // QLoRA 默认
    int32_t num_groups = (num_blocks + group_size - 1) / group_size;

    // 2. 分配并读取数据
    std::vector<uint8_t> packed_weights((total_elements + 1) / 2);
    std::vector<uint8_t> absmax_q(num_blocks);
    std::vector<uint16_t> code2(256);
    std::vector<uint16_t> absmax2(num_groups);
    float offset;

    ifs.read(reinterpret_cast<char*>(packed_weights.data()), packed_weights.size());
    ifs.read(reinterpret_cast<char*>(absmax_q.data()), absmax_q.size());
    ifs.read(reinterpret_cast<char*>(code2.data()), code2.size() * 2);
    ifs.read(reinterpret_cast<char*>(absmax2.data()), absmax2.size() * 2);
    ifs.read(reinterpret_cast<char*>(&offset), 4);

    // 3. CPU 反量化逻辑 (原型)
    std::vector<float> output(total_elements);
    for (int64_t i = 0; i < static_cast<int64_t>(packed_weights.size()); ++i) {
        uint8_t byte = packed_weights[i];
        
        // 拆解两个 4-bit 索引
        uint8_t idxs[2];
        idxs[0] = byte & 0x0F;         // 低4位
        idxs[1] = (byte >> 4) & 0x0F;  // 高4位

        for (int j = 0; j < 2; ++j) {
            int64_t curr_idx = i * 2 + j;
            if (curr_idx >= total_elements) {
                continue;
            }
            int32_t b_idx = curr_idx / header.blocksize;
            int32_t g_idx = b_idx / group_size;

            // 双重解量化公式
            float s1 = half_to_float_cpu(code2[absmax_q[b_idx]]);
            float s2 = half_to_float_cpu(absmax2[g_idx]);
            float scale = s1 * s2;

            output[curr_idx] = NF4_TABLE[idxs[j]] * scale + offset;
        }
    }

    // 4. 验证结果 (读取 gt_output.bin)
    std::vector<uint16_t> gt(total_elements);
    std::ifstream gfs("gt_output.bin", std::ios::binary);
    gfs.read(reinterpret_cast<char*>(gt.data()), total_elements * 2);

    float max_error = 0;
    for(int i=0; i<total_elements; ++i) {
        max_error = std::max(max_error, std::abs(output[i] - half_to_float_cpu(gt[i])));
    }
    std::cout << "Max Absolute Error: " << max_error << std::endl;

    return 0;
}