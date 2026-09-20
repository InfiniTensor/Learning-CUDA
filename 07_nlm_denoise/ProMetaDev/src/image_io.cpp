// image_io.cpp - 图像读写（OpenCV）与质量指标实现
#include "image_io.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace nlm {

namespace {
double now_ms() {
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}
std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos)
        return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}
} // namespace

Image load_image(const std::string& path) {
    cv::Mat m = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (m.empty())
        throw std::runtime_error("无法读取图像（不支持或文件不存在）: " + path);

    Image img;
    img.width = m.cols;
    img.height = m.rows;

    // 只处理 8 位；通道数保留 1（灰度）或取前 3（RGB）
    cv::Mat mm;
    if (m.depth() != CV_8U)
        m.convertTo(mm, CV_8U);
    else
        mm = m;
    if (mm.channels() == 2) {
        std::vector<cv::Mat> ch;
        cv::split(mm, ch);
        mm = ch[0];
    }
    img.channels = std::min(mm.channels(), 3);
    if (img.channels != 1 && img.channels != 3)
        throw std::runtime_error("仅支持灰度(1 通道)或 RGB(3 通道)图像");

    cv::Mat cont = mm.isContinuous() ? mm : mm.clone();
    // 通道顺序沿用 OpenCV 约定（读写一致即可；NLM 对通道是对称处理的）
    if (img.channels == 1) {
        cv::Mat g;
        cv::extractChannel(cont, g, 0);
        img.data.assign(g.data, g.data + (size_t)g.total());
    } else {
        cv::Mat rgb;
        if (cont.channels() == 3) {
            rgb = cont;
        } else { // 4 通道：丢弃 alpha
            std::vector<cv::Mat> ch;
            cv::split(cont, ch);
            ch.resize(3);
            cv::merge(ch, rgb);
        }
        img.data.assign(rgb.data, rgb.data + (size_t)img.pixels() * 3);
    }
    return img;
}

void save_image(const std::string& path, const Image& img) {
    if (img.empty())
        throw std::runtime_error("待保存的图像为空");
    cv::Mat m(img.height, img.width, img.channels == 1 ? CV_8UC1 : CV_8UC3,
              const_cast<uint8_t*>(img.data.data()));
    if (!cv::imwrite(path, m))
        throw std::runtime_error("无法写入图像: " + path);
}

NLMParams load_params(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("无法打开参数文件: " + path);

    NLMParams p;
    std::string line;
    while (std::getline(in, line)) {
        size_t h = line.find_first_not_of(" \t");
        if (h == std::string::npos || line[h] == '#')
            continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos)
            continue;
        std::string k = trim(line.substr(0, eq));
        std::string v = trim(line.substr(eq + 1));
        // 去掉行尾注释
        size_t c = v.find('#');
        if (c != std::string::npos)
            v = trim(v.substr(0, c));
        if (k == "patch_radius")
            p.patch_radius = std::stoi(v);
        else if (k == "search_radius")
            p.search_radius = std::stoi(v);
        else if (k == "h")
            p.h = std::stod(v);
        else if (k == "sigma")
            p.sigma = std::stod(v);
        else if (k == "search_step")
            p.search_step = std::stoi(v);
        else if (k == "patch_step")
            p.patch_step = std::stoi(v);
        else if (k == "use_lut")
            p.use_lut = (v == "1" || v == "true");
        else if (k == "lut_bins")
            p.lut_bins = std::stoi(v);
    }
    if (p.patch_radius < 1)
        throw std::runtime_error("patch_radius 必须 >= 1");
    if (p.search_radius < 1)
        throw std::runtime_error("search_radius 必须 >= 1");
    if (p.h <= 0.0)
        throw std::runtime_error("h 必须为正");
    if (p.sigma < 0.0)
        throw std::runtime_error("sigma 必须非负");
    if (p.search_step < 1 || p.patch_step < 1)
        throw std::runtime_error("search_step / patch_step 必须 >= 1");
    return p;
}

Quality compare(const Image& a, const Image& b) {
    if (a.width != b.width || a.height != b.height || a.channels != b.channels)
        throw std::runtime_error("compare: 两图尺寸/通道不一致");
    Quality q;
    double sum = 0.0, sq = 0.0, mx = 0.0;
    const size_t n = a.data.size();
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double)a.data[i] - (double)b.data[i]);
        sum += d;
        sq += d * d;
        mx = std::max(mx, d);
    }
    q.mae = sum / (double)n;
    q.mse = sq / (double)n;
    q.max_abs = mx;
    q.psnr = (q.mse > 1e-12) ? 10.0 * std::log10(255.0 * 255.0 / q.mse) : 99.0;
    return q;
}

} // namespace nlm
