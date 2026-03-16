#ifndef BILATERAL_FILTER_OPENCV_H_
#define BILATERAL_FILTER_OPENCV_H_

#include "image_io.h"

void apply_bilateral_filter_opencv(const ImageData& input, ImageData& output,
                                   const FilterParams& params);

double compute_mae(const ImageData& img1, const ImageData& img2);
double compute_psnr(const ImageData& img1, const ImageData& img2);

#ifdef HAVE_OPENCV_CUDA
// OpenCV CUDA bilateral filter (cv::cuda::bilateralFilter).
// Returns false if CUDA device is unavailable at runtime.
bool apply_bilateral_filter_opencv_cuda(const ImageData& input, ImageData& output,
                                        const FilterParams& params);
#endif

#endif // BILATERAL_FILTER_OPENCV_H_
