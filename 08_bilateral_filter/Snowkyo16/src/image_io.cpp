#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image_io.h"
#include <iostream>

using namespace std;

// 加载图像文件
Image load_image(const string& filepath) {
    Image img;

    // stbi_load 参数：文件路径、宽/高/通道数的指针、期望通道数（0=原始）
    uint8_t* raw = stbi_load(filepath.c_str(),
                             &img.width,
                             &img.height,
                             &img.channels,
                             0); 

    if (!raw) {
        cerr << "错误：无法加载图像 " << filepath
             << " (" << stbi_failure_reason() << ")" << endl;
        exit(EXIT_FAILURE);
    }

    // 总字节数 = 宽 × 高 × 通道数
    size_t total_bytes = (size_t)img.width * img.height * img.channels;

    // 将原始指针数据拷贝到 vector 中（自动管理内存）
    img.data.assign(raw, raw + total_bytes);

    // 释放 stb 分配的原始内存
    stbi_image_free(raw);

    cout << "加载图像: " << filepath
         << " (" << img.width << "x" << img.height
         << ", " << img.channels << " 通道)" << endl;

    return img;
}

void save_image(const string& filepath, const Image& img) {
    int stride = img.width * img.channels;
    int ret = stbi_write_png(filepath.c_str(),
                             img.width,
                             img.height,
                             img.channels,
                             img.data.data(),  // 像素数据指针
                             stride);  // 每行字节数（width * channels）

    if (!ret) {
        cerr << "错误：无法保存图像 " << filepath << endl;
        exit(EXIT_FAILURE);
    }

    cout << "保存图像: " << filepath << endl;
}
