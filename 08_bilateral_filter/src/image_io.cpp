#include "image_io.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

bool load_raw_image(const char* path, ImageData* img) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open image file: %s\n", path);
        return false;
    }

    if (fread(&img->width, sizeof(int), 1, fp) != 1 ||
        fread(&img->height, sizeof(int), 1, fp) != 1 ||
        fread(&img->channels, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Failed to read image header\n");
        fclose(fp);
        return false;
    }

    if (img->width <= 0 || img->height <= 0 || img->channels <= 0 || img->channels > 4) {
        fprintf(stderr, "Invalid image dimensions: %dx%d, channels=%d\n",
                img->width, img->height, img->channels);
        fclose(fp);
        return false;
    }

    size_t data_size = img->size();
    img->data.resize(data_size);

    if (fread(img->data.data(), sizeof(uint8_t), data_size, fp) != data_size) {
        fprintf(stderr, "Failed to read image data\n");
        fclose(fp);
        return false;
    }

    fclose(fp);
    return true;
}

bool save_raw_image(const char* path, const ImageData& img) {
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to create output file: %s\n", path);
        return false;
    }

    fwrite(&img.width, sizeof(int), 1, fp);
    fwrite(&img.height, sizeof(int), 1, fp);
    fwrite(&img.channels, sizeof(int), 1, fp);
    fwrite(img.data.data(), sizeof(uint8_t), img.size(), fp);

    fclose(fp);
    return true;
}

bool load_params(const char* path, FilterParams* params) {
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open params file: %s\n", path);
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;

        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);

        auto trim = [](std::string& s) {
            size_t start = s.find_first_not_of(" \t\r\n");
            size_t end = s.find_last_not_of(" \t\r\n#");
            if (start != std::string::npos && end != std::string::npos) {
                s = s.substr(start, end - start + 1);
            }
        };
        trim(key);
        trim(value);

        if (key == "radius") {
            params->radius = std::stoi(value);
        } else if (key == "sigma_spatial") {
            params->sigma_spatial = std::stof(value);
        } else if (key == "sigma_color") {
            params->sigma_color = std::stof(value);
        }
    }

    if (params->radius <= 0 || params->sigma_spatial <= 0 || params->sigma_color <= 0) {
        fprintf(stderr, "Invalid parameters: radius=%d, sigma_spatial=%.2f, sigma_color=%.2f\n",
                params->radius, params->sigma_spatial, params->sigma_color);
        return false;
    }

    return true;
}
