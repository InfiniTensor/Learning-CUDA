#include "image_io.h"
#include "params.h"
#include "bilateral_filter.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>

using namespace std;

// 打印性能日志
void print_performance(double time_ms, int width, int height) {
    double megapixels = (double)(width * height) / 1e6;
    double throughput = megapixels / (time_ms / 1000.0);
    double fps = 1000.0 / time_ms;

    cout << fixed << setprecision(2);
    cout << "  处理时间:      " << time_ms << " ms" << endl;
    cout << "  吞吐量:        " << throughput << " MPixels/s" << endl;
    cout << "  FPS:           " << fps << endl;
    cout << endl;
}


// 提取文件名（不含目录）
string get_filename(const string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == string::npos) return path;
    return path.substr(pos + 1);
}

// V0 —— CPU 基线
int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "用法: " << argv[0]
             << " <输入图像> <参数文件> <输出目录>" << endl;
        return 1;
    }

    string input_path  = argv[1];
    string param_path  = argv[2];
    string output_dir  = argv[3];

    // 构造输出文件路径
    string filename = get_filename(input_path);
    string basename = filename;
    size_t dot_pos = basename.rfind('.');
    string ext = ".png";
    if (dot_pos != string::npos) {
        ext = basename.substr(dot_pos);
        basename = basename.substr(0, dot_pos);
    }

    // 加载输入
    cout << "========================================" << endl;
    cout << "           双边滤波 V0 - CPU基线          " << endl;
    cout << "========================================" << endl;
    cout << endl;

    Image input = load_image(input_path);

    FilterParams params = load_params(param_path);
    cout << endl;
    cout << "  [参数]" << endl;
    cout << "  图像尺寸:      " << input.width << " x " << input.height
         << " (" << input.channels << " 通道)" << endl;
    cout << "  窗口半径:      " << params.radius
         << " (窗口大小 " << 2 * params.radius + 1
         << "x" << 2 * params.radius + 1 << ")" << endl;
    cout << "  sigma_spatial: " << params.sigma_spatial << endl;
    cout << "  sigma_color:   " << params.sigma_color << endl;

    cout << endl;
    cout << "  [CPU 双边滤波]" << endl;

    // 计算时间差
    auto t0 = chrono::high_resolution_clock::now();
    Image cpu_output = bilateral_filter_cpu(input, params);
    auto t1 = chrono::high_resolution_clock::now();

    double cpu_ms = chrono::duration<double, milli>(t1 - t0).count();
    print_performance(cpu_ms, input.width, input.height);

    // 保存CPU结果到输出目录
    string image_dir = output_dir + "/images";
    string cpu_out_path = image_dir + "/" + basename + "_cpu" + ext;
    save_image(cpu_out_path, cpu_output);

    // 性能目标对比
    double target_fps = 60.0;
    double target_ms = 1000.0 / target_fps;

    cout << endl;
    cout << "  [4K 60fps 目标差距]" << endl;
    cout << fixed << setprecision(2);
    cout << "  目标帧耗时:    " << target_ms << " ms" << endl;
    cout << "  当前帧耗时:    " << cpu_ms << " ms" << endl;
    cout << "  差距倍数:      " << cpu_ms / target_ms << "x" << endl;

    cout << endl;
    cout << "========================================" << endl;

    return 0;
}
