#include "bilateral_filter_opencv.h"

#include <cmath>
#include <opencv2/opencv.hpp>

#ifdef HAVE_OPENCV_CUDA
#include <opencv2/cudaimgproc.hpp>
#endif

void apply_bilateral_filter_opencv(const ImageData& input, ImageData& output,
                                   const FilterParams& params) {
    int cv_type = input.channels == 1 ? CV_8UC1 : CV_8UC3;
    cv::Mat src(input.height, input.width, cv_type, const_cast<uint8_t*>(input.data.data()));
    cv::Mat dst;

    int d = 2 * params.radius + 1;
    cv::bilateralFilter(src, dst, d, params.sigma_color, params.sigma_spatial);

    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.data.resize(output.size());
    std::memcpy(output.data.data(), dst.data, output.size());
}

#ifdef HAVE_OPENCV_CUDA
bool apply_bilateral_filter_opencv_cuda(const ImageData& input, ImageData& output,
                                        const FilterParams& params) {
    // Check CUDA device availability at runtime
    if (cv::cuda::getCudaEnabledDeviceCount() <= 0) {
        return false;
    }

    int cv_type = input.channels == 1 ? CV_8UC1 : CV_8UC3;
    cv::Mat src(input.height, input.width, cv_type, const_cast<uint8_t*>(input.data.data()));

    // Upload to GPU
    cv::cuda::GpuMat d_src, d_dst;
    d_src.upload(src);

    int kernel_size = 2 * params.radius + 1;
    cv::cuda::bilateralFilter(d_src, d_dst, kernel_size, params.sigma_color, params.sigma_spatial);

    // Download result
    cv::Mat dst;
    d_dst.download(dst);

    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.data.resize(output.size());
    std::memcpy(output.data.data(), dst.data, output.size());
    return true;
}
#endif

double compute_mae(const ImageData& img1, const ImageData& img2) {
    if (img1.width != img2.width || img1.height != img2.height || img1.channels != img2.channels) {
        return -1.0;
    }

    double sum = 0.0;
    size_t count = img1.size();

    for (size_t i = 0; i < count; ++i) {
        double diff =
            std::abs(static_cast<double>(img1.data[i]) - static_cast<double>(img2.data[i]));
        sum += diff;
    }

    return sum / count;
}

double compute_psnr(const ImageData& img1, const ImageData& img2) {
    if (img1.width != img2.width || img1.height != img2.height || img1.channels != img2.channels) {
        return -1.0;
    }

    double sum_sq = 0.0;
    size_t count = img1.size();

    for (size_t i = 0; i < count; ++i) {
        double diff = static_cast<double>(img1.data[i]) - static_cast<double>(img2.data[i]);
        sum_sq += diff * diff;
    }

    double mse = sum_sq / count;
    if (mse < 1e-10) {
        return 999.99; // practically identical
    }
    // MAX = 255 for uint8
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}
