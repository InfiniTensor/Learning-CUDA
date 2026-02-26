#ifndef BILATERAL_FILTER_OPENCV_H_
#define BILATERAL_FILTER_OPENCV_H_

#include "image_io.h"

void apply_bilateral_filter_opencv(const ImageData& input,
                                   ImageData& output,
                                   const FilterParams& params);

double compute_mae(const ImageData& img1, const ImageData& img2);

#endif  // BILATERAL_FILTER_OPENCV_H_
