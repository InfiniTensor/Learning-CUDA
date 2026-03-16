//
// Created by flashzxi on 2/24/26.
//
#include "quant_state.h"
#include "cuda_runtime.h"
#include "nf4_dequant.h"

// https://gxtctab8no8.feishu.cn/wiki/UoESwCDZ2iZRcLkdzjvcxTgenOb?from=from_copylink

int main() {
    int row = 10000;
    int col = 10000;
    std::string file_prefix = std::string("/home/core_dump/Learning-CUDA/03_nf4_dequant/flashzxi/nf4_") + std::to_string(row) + "x" + std::to_string(col) + "_fp16";
    auto conf = parse_quant_state(file_prefix  + ".bin",
        "/home/core_dump/Learning-CUDA/03_nf4_dequant/flashzxi/test/conf/blocksize64_fp16_T4.ini",
        file_prefix + "_w_dequant.bin");

    // conf.print();

    // std::cout << "real absmax: ";
    // for (int i = 0; i < 4; i ++) {
    //     int idx = conf.absmax_q[i];
    //     std::cout << __half2float(conf.code2[idx] * conf.absmax2[0]) + conf.offset << " ";
    // }

    std::cout << std::endl;
    __half* ans = new __half[conf.num_elements];
    nf4_dequant_warp8_batch8_one_phase(conf, ans);

    float max_diff = 0.f;
    for (int i = 0; i < conf.num_rows; i++) {
        for (int j = 0; j < conf.num_cols; j++) {
            int idx = i * conf.num_cols + j;
            float a = __half2float(ans[idx]);
            float b = __half2float(conf.ref_result[idx]);
            float diff = fabsf(a - b);
            diff /= b;
            max_diff = std::max(max_diff, diff);
            // std::cout << a << " ";
        }
        // std::cout << "\n";
    }
    std::cout << "max_diff = " << max_diff << "\n";
}

