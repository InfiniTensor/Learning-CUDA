#ifndef BILATERAL_FILTER_H_
#define BILATERAL_FILTER_H_

#include "image_io.h"
#include <vector>

void bilateral_filter_cpu(const float* input,
                          float* output,
                          int width,
                          int height,
                          int channels,
                          int radius,
                          float sigma_spatial,
                          float sigma_color);

void apply_bilateral_filter_cpu(const ImageData& input,
                                ImageData& output,
                                const FilterParams& params);

#endif  // BILATERAL_FILTER_H_
