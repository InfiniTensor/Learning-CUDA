// ============================================================================
// metrics 子命令：对同尺寸图像计算 MAE/PSNR，并可追加写入结构化 CSV。
// 该路径只读取图像，不依赖 GPU，便于在服务器与开发机之间复核质量结果。
// ============================================================================
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>

#include "image_io.h"
#include "benchmark.h"

namespace {

bool FileHasContent(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary | std::ios::ate);
    return in && in.tellg() > 0;
}

}  // namespace

int RunMetrics(const std::string& reference_path, const std::string& test_path,
               const std::string& label, const std::string& csv_path) {
    std::string err;
    ImageU8 reference;
    ImageU8 test;
    if (!LoadImage(reference_path.c_str(), &reference, &err)) {
        std::fprintf(stderr, "[metrics] reference: %s\n", err.c_str());
        return 2;
    }
    if (!LoadImage(test_path.c_str(), &test, &err)) {
        std::fprintf(stderr, "[metrics] test: %s\n", err.c_str());
        return 2;
    }
    if (reference.width != test.width || reference.height != test.height ||
        reference.channels != test.channels) {
        std::fprintf(stderr,
                     "[metrics] 尺寸/通道不一致: reference=%dx%dx%d test=%dx%dx%d\n",
                     reference.width, reference.height, reference.channels,
                     test.width, test.height, test.channels);
        return 2;
    }

    const double mae = ComputeMAE(reference, test);
    const double psnr = ComputePSNR(reference, test);
    std::printf("[metrics] %s  %dx%dx%d  MAE=%.6f  PSNR=%.6f dB\n",
                label.c_str(), reference.width, reference.height, reference.channels,
                mae, psnr);

    if (!csv_path.empty()) {
        const bool has_content = FileHasContent(csv_path);
        EnsureParentDir(csv_path.c_str());
        std::ofstream csv(csv_path.c_str(), std::ios::app);
        if (!csv) {
            std::fprintf(stderr, "[metrics] 无法写入 CSV: %s\n", csv_path.c_str());
            return 2;
        }
        if (!has_content) {
            csv << "label,reference,test,width,height,channels,mae,psnr_db\n";
        }
        csv << label << ',' << reference_path << ',' << test_path << ','
            << reference.width << ',' << reference.height << ',' << reference.channels << ','
            << std::fixed << std::setprecision(6) << mae << ',' << psnr << '\n';
    }
    return 0;
}
