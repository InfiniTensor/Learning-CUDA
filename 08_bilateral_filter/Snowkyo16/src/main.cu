#include "image_io.h"
#include "params.h"
#include "bilateral_filter.cuh"
#include "benchmark.h"

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// 提取文件名（不含目录）
string get_filename(const string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == string::npos) return path;
    return path.substr(pos + 1);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "用法: " << argv[0]
             << " <输入图像> <参数文件> <输出目录> [版本]" << endl;
        cerr << "版本: v0 / v1/ v2/ v3/ all (默认 all) " << endl;
        return 1;
    }

    string input_path = argv[1];
    string param_path = argv[2];
    string output_dir = argv[3];
    string mode = (argc > 4) ? argv[4] : "all";

    // 解析文件名
    string filename = get_filename(input_path);
    string basename = filename;
    size_t dot_pos = basename.rfind('.');
    // 输出统一用 .png(无损)，避免JPEG压缩引入误差
    if (dot_pos != string::npos) {
        basename = basename.substr(0, dot_pos);
    }
    string ext = ".png";
    string image_dir = output_dir + "/images";

    // 加载输入
    cout << "========================================" << endl;
    cout << "            双边滤波 - 运行模式:           " << endl;
    cout << "========================================" << endl;
    cout << endl;

    Image input = load_image(input_path);
    FilterParams params = load_params(param_path);

    // 首次CUDA调用会触发上下文初始化，提前做一次，避免计入V1的计时
    cudaFree(0);

    cout << endl;
    cout << "  [参数]" << endl;
    cout << "  图像尺寸:      " << input.width << " x " << input.height
         << " (" << input.channels << " 通道)" << endl;
    cout << "  窗口半径:      " << params.radius
         << " (窗口大小 " << 2 * params.radius + 1
         << "x" << 2 * params.radius + 1 << ")" << endl;
    cout << "  sigma_spatial: " << params.sigma_spatial << endl;
    cout << "  sigma_color:   " << params.sigma_color << endl;

    // 按版本执行
    vector<VersionResult> results;
    int gpu_runs = 10;  // GPU版本：1次预热 + 10次计时取平均

    if (mode == "v0" || mode == "all") {
        results.push_back(run_version("v0_cpu", input, params, 
                                      bilateral_filter_cpu_v0, image_dir, 
                                      basename, ext));
    }

    if (mode == "v1" || mode == "all") {
        results.push_back(run_version("v1_naive", input, params,
                                      bilateral_filter_gpu_v1, image_dir, 
                                      basename, ext, gpu_runs));
    }

    if (mode == "v2" || mode == "all") {
        results.push_back(run_version("v2_smem", input, params,
                                      bilateral_filter_gpu_v2, image_dir,
                                      basename, ext, gpu_runs));
    }

    if (mode == "v3" || mode == "all") {
        bilateral_filter_gpu_v3_init(params);  // LUT上传，不计入每帧耗时
        results.push_back(run_version("v3_constmem", input, params,
                                      bilateral_filter_gpu_v3, image_dir,
                                      basename, ext, gpu_runs));
    }

    print_summary(results, input.width, input.height);

    cout << endl;
    cout << "========================================" << endl;

    return 0;
}
