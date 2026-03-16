#ifndef IMAGE_IO_H_
#define IMAGE_IO_H_

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct ImageData {
    int width;
    int height;
    int channels;
    std::vector<uint8_t> data;

    size_t size() const { return static_cast<size_t>(width) * height * channels; }
    bool valid() const { return width > 0 && height > 0 && channels > 0 && !data.empty(); }
};

struct FilterParams {
    int radius = 0;
    float sigma_spatial = 0.0f;
    float sigma_color = 0.0f;
};

bool load_raw_image(const char* path, ImageData* img);
bool save_raw_image(const char* path, const ImageData& img);
bool load_params(const char* path, FilterParams* params);

#endif // IMAGE_IO_H_
