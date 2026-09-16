#ifndef NLM_IMAGE_IO_H
#define NLM_IMAGE_IO_H

#include <cstdint>
#include <string>
#include <vector>

// ============================================================================
// 图像对象（Host 侧，interleaved 布局，与文件格式一致）
//   channels == 1 : 灰度图，data[y*w + x]
//   channels == 3 : RGB 图， data[(y*w + x)*3 + ch]
// 像素值范围 [0,255]（8 位整数），处理时由转换 kernel 转为浮点。
// ============================================================================
struct ImageU8 {
    int width;
    int height;
    int channels;                 // ∈ {1, 3}
    std::vector<uint8_t> data;

    ImageU8() : width(0), height(0), channels(0) {}
};

// 解码 PNG/JPG/BMP/TGA；仅接受灰度或 RGB（带 alpha 的文件自动丢弃 alpha 通道）
bool LoadImage(const char* path, ImageU8* out, std::string* err);

// 按输出扩展名编码（.png/.jpg/.jpeg/.bmp/.tga，缺省 PNG），与输入同格式语义
// 保存前自动逐级创建父目录（如 output/images/ 不存在时创建），便于集中管理数据
bool SaveImage(const char* path, const ImageU8& img, std::string* err);

// 确保路径的父目录存在（逐级创建，已存在则跳过；相对/无目录路径直接成功）
// 数据集中管理配套：日志等非图像输出路径同样可用（POSIX mkdir，Linux 部署）
void EnsureParentDir(const char* path);

#endif // NLM_IMAGE_IO_H
