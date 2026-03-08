#include "benchmark.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

using namespace std;

// ============================================================
// 运行一个版本：计时 + 打印性能 + 保存输出图片
// ============================================================
VersionResult run_version(const string& name, const Image& input, 
                          const FilterParams& params, FilterFunc func,
                          const string& image_dir, const string& basename,
                          const string& ext) {
    cout << endl;
    cout << "  [" << name << "]" << endl;

    // 计时
    auto t0 = chrono::high_resolution_clock::now();
    Image output = func(input, params);
    auto t1 = chrono::high_resolution_clock::now();

    double ms = chrono::duration<double, milli>(t1 - t0).count();
    double megapixels = (double)(input.width * input.height) / 1e6;
    double throughput = megapixels / (ms / 1000.0);
    double fps = 1000.0 / ms;

    cout << fixed << setprecision(2);
    cout << "  处理时间:      " << ms << " ms" << endl;
    cout << "  吞吐量:        " << throughput << " MPixels/s" << endl;
    cout << "  FPS:           " << fps << " fps" << endl;

    // 保存图片
    string out_path = image_dir + "/" + basename + "_" + name + ext;
    save_image(out_path, output);

    return {name, ms, move(output)};
}

// 打印汇总对比表 
void print_summary(const vector<VersionResult>& results, int width, int height) {
    if (results.empty()) return;

    double megapixels = (double)(width * height) / 1e6;
    double base_ms = results[0].time_ms;

    // 一致性验证（所有版本与第一个版本对比）
    if (results.size() >= 2) {
        cout << endl;
        cout << "  [一致性验证 (vs " << results[0].name << ")]" << endl;

        for (size_t i = 1; i < results.size(); i++) {
            int max_diff = 0;
            for (size_t j = 0; j < results[0].output.data.size(); j++) {
                int diff = abs((int)results[0].output.data[j] -
                               (int)results[i].output.data[j]);
                if (diff > max_diff) max_diff = diff;
            }
            cout << "  " << results[i].name << ": 最大像素差异 = " << max_diff;
            if (max_diff == 0) cout << " (完全一致)";
            else if (max_diff <= 1) cout << " (可接受)";
            else cout << " (请检查！)";
            cout << endl;
        }
    }

    // 性能对比表
    cout << endl;
    cout << "  [性能对比]" << endl;
    cout << fixed << setprecision(2);
    cout << "  " << string(50, '-') << endl;
    cout << "  " << left << setw(16) << "版本"
         << right << setw(12) << "耗时(ms)"
         << setw(16) << "吞吐(MP/s)"
         << setw(13) << "加速比" << endl;
    cout << "  " << string(50, '-') << endl;

    for (const auto& r : results) {
        double throughput = megapixels / (r.time_ms / 1000.0);
        double speedup = base_ms / r.time_ms;
        cout << "  " << left << setw(15) << r.name
             << right << setw(9) << r.time_ms
             << setw(13) << throughput
             << setw(10) << speedup << "x" << endl;
    }
    cout << "  " << string(50, '-') << endl;

    // 4K 60fps 目标 (基于吞吐量对比，与图像尺寸无关)
    // 4K = 3840x2160 = 8.29 MPixels, 60fps需要497.66MPixels/s
    double target_throughput = 3840.0 * 2160.0 / 1e6 * 60.0;
    double best_ms = results.back().time_ms;
    double best_throughput = megapixels / (best_ms / 1000.0);
    cout << endl;
    cout << "  [4K 60fps 目标] " << endl;
    cout << "  目标吞吐量:      " << target_throughput << " MPixels/s" << endl;
    cout << "  当前最快吞吐量:  " << best_throughput << " MPixels/s" << endl;
    if (best_throughput >= target_throughput) {
        cout << "  状态: 已达标！  " << endl;
    } else {
        cout << "  还需提升:        " << target_throughput / best_throughput << "x" << endl;
    }
}
