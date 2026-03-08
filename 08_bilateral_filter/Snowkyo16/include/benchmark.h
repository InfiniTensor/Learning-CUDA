#pragma once

#include "image_io.h"
#include "params.h"

#include <string>
#include <vector>
#include <functional>

using namespace std;

// 滤波函数统一签名（CPU 和 GPU 版本共用）
using FilterFunc = function<Image(const Image&, const FilterParams&)>;

// 单个版本的运行结果
struct VersionResult {
    string name;        // 如 "v0_cpu", "v1_naive"
    double time_ms;
    Image output;
};

// 运行一个版本：计时 + 打印 + 保存图片，返回结果
VersionResult run_version(const string& name, const Image& input, 
                          const FilterParams& params, FilterFunc func, 
                          const string& image_dir, const string& basename, 
                          const string& ext);

// 打印汇总对比表 + 一致性验证
void print_summary(const vector<VersionResult>& results, int width, int height);
