#include "image_io.h"

#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>

// stb_image / stb_image_write：单头文件图像编解码库（零外部依赖，便于多平台移植）
#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

std::string LowerExt(const char* path) {
    std::string p(path);
    size_t dot = p.find_last_of('.');
    if (dot == std::string::npos) return "";
    std::string ext = p.substr(dot + 1);
    for (size_t i = 0; i < ext.size(); ++i)
        if (ext[i] >= 'A' && ext[i] <= 'Z') ext[i] = (char)(ext[i] - 'A' + 'a');
    return ext;
}

} // namespace

void EnsureParentDir(const char* path) {
    if (!path) return;
    const std::string p(path);
    const size_t slash = p.find_last_of('/');
    if (slash == std::string::npos || slash == 0) return;  // 无目录或根目录
    // 逐级创建（已存在时 mkdir 报 EEXIST，直接忽略）
    for (size_t i = 1; i < slash; ++i) {
        if (p[i] == '/') {
            std::string part = p.substr(0, i);
            mkdir(part.c_str(), 0755);
        }
    }
    mkdir(p.substr(0, slash).c_str(), 0755);
}

bool LoadImage(const char* path, ImageU8* out, std::string* err) {
    if (!path || !out) {
        if (err) *err = "LoadImage: 空指针参数";
        return false;
    }
    int w = 0, h = 0, c = 0;
    unsigned char* raw = stbi_load(path, &w, &h, &c, 0);
    if (!raw) {
        if (err) *err = std::string("图像解码失败: ") + path + " (" + stbi_failure_reason() + ")";
        return false;
    }
    // 灰度+alpha / RGBA：重新解码并丢弃 alpha，保证通道语义 ∈ {1,3}
    if (c == 2 || c == 4) {
        stbi_image_free(raw);
        const int want = (c == 2) ? 1 : 3;
        raw = stbi_load(path, &w, &h, &c, want);
        if (!raw) {
            if (err) *err = std::string("图像解码失败(重试): ") + path;
            return false;
        }
        c = want;
    }
    if (c != 1 && c != 3) {
        stbi_image_free(raw);
        if (err) *err = "仅支持灰度或 RGB 图像，实际通道数: " + std::to_string(c);
        return false;
    }

    out->width = w;
    out->height = h;
    out->channels = c;
    out->data.assign(raw, raw + (size_t)w * h * c);
    stbi_image_free(raw);
    return true;
}

bool SaveImage(const char* path, const ImageU8& img, std::string* err) {
    if (!path) {
        if (err) *err = "SaveImage: 空路径";
        return false;
    }
    if (img.width <= 0 || img.height <= 0 ||
        (img.channels != 1 && img.channels != 3) ||
        img.data.size() != (size_t)img.width * img.height * img.channels) {
        if (err) *err = "SaveImage: 非法图像对象";
        return false;
    }
    EnsureParentDir(path);  // 数据集中管理：输出目录不存在时自动创建
    const std::string ext = LowerExt(path);
    int ok = 0;
    if (ext == "jpg" || ext == "jpeg") {
        ok = stbi_write_jpg(path, img.width, img.height, img.channels, img.data.data(), 95);
    } else if (ext == "bmp") {
        ok = stbi_write_bmp(path, img.width, img.height, img.channels, img.data.data());
    } else if (ext == "tga") {
        ok = stbi_write_tga(path, img.width, img.height, img.channels, img.data.data());
    } else {
        // 缺省 / .png
        ok = stbi_write_png(path, img.width, img.height, img.channels,
                            img.data.data(), img.width * img.channels);
    }
    if (!ok) {
        if (err) *err = std::string("图像编码失败: ") + path;
        return false;
    }
    return true;
}
