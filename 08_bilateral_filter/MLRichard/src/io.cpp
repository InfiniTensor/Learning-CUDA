#include "io.h"
#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>

Image read_raw(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    Image img;
    f.read(reinterpret_cast<char*>(&img.width),    4);
    f.read(reinterpret_cast<char*>(&img.height),   4);
    f.read(reinterpret_cast<char*>(&img.channels), 4);
    size_t n = (size_t)img.width * img.height * img.channels;
    img.data.resize(n);
    f.read(reinterpret_cast<char*>(img.data.data()), n);
    return img;
}

void write_raw(const std::string& path, const Image& img) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write: " + path);
    f.write(reinterpret_cast<const char*>(&img.width),    4);
    f.write(reinterpret_cast<const char*>(&img.height),   4);
    f.write(reinterpret_cast<const char*>(&img.channels), 4);
    f.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
}

Image load_image(const std::string& path) {
    cv::Mat mat = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (mat.empty()) throw std::runtime_error("Cannot load image: " + path);
    Image img;
    img.width    = mat.cols;
    img.height   = mat.rows;
    img.channels = mat.channels();
    if (img.channels == 3) cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
    img.data.assign(mat.data, mat.data + mat.total() * mat.elemSize());
    return img;
}

Image float_to_image(const float* data, uint32_t w, uint32_t h, uint32_t c) {
    Image img;
    img.width = w; img.height = h; img.channels = c;
    size_t n = (size_t)w * h * c;
    img.data.resize(n);
    for (size_t i = 0; i < n; i++) {
        float v = data[i];
        if (v < 0.f) v = 0.f;
        if (v > 255.f) v = 255.f;
        img.data[i] = static_cast<uint8_t>(v + 0.5f);
    }
    return img;
}

std::vector<float> image_to_float(const Image& img) {
    size_t n = (size_t)img.width * img.height * img.channels;
    std::vector<float> f(n);
    for (size_t i = 0; i < n; i++) f[i] = static_cast<float>(img.data[i]);
    return f;
}
