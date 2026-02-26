#include <algorithm>
#include <cmath>

#include "bilateral_filter.h"

void bilateral_filter_cpu(const float* input, float* output, int width, int height, int channels,
                          int radius, float sigma_spatial, float sigma_color) {
    float spatial_coeff = -0.5f / (sigma_spatial * sigma_spatial);
    float color_coeff = -0.5f / (sigma_color * sigma_color);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int center_idx = (y * width + x) * channels;

            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                float weight_sum = 0.0f;
                float center_val = input[center_idx + c];

                for (int dy = -radius; dy <= radius; ++dy) {
                    int ny = y + dy;
                    if (ny < 0 || ny >= height)
                        continue;

                    for (int dx = -radius; dx <= radius; ++dx) {
                        int nx = x + dx;
                        if (nx < 0 || nx >= width)
                            continue;

                        int neighbor_idx = (ny * width + nx) * channels;
                        float neighbor_val = input[neighbor_idx + c];

                        float spatial_dist = static_cast<float>(dx * dx + dy * dy);
                        float spatial_weight = expf(spatial_dist * spatial_coeff);

                        float color_dist = neighbor_val - center_val;
                        float color_weight = expf(color_dist * color_dist * color_coeff);

                        float weight = spatial_weight * color_weight;
                        sum += neighbor_val * weight;
                        weight_sum += weight;
                    }
                }

                output[center_idx + c] = sum / weight_sum;
            }
        }
    }
}

void apply_bilateral_filter_cpu(const ImageData& input, ImageData& output,
                                const FilterParams& params) {
    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.data.resize(input.size());

    std::vector<float> input_float(input.size());
    std::vector<float> output_float(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        input_float[i] = static_cast<float>(input.data[i]);
    }

    bilateral_filter_cpu(input_float.data(), output_float.data(), input.width, input.height,
                         input.channels, params.radius, params.sigma_spatial, params.sigma_color);

    for (size_t i = 0; i < output.size(); ++i) {
        output.data[i] = static_cast<uint8_t>(std::clamp(output_float[i], 0.0f, 255.0f));
    }
}
