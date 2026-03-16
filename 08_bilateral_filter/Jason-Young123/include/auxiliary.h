//文件读写等辅助函数
#pragma once

#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <myPixel.h>


bool getCfg(const std::string& cfgPath, int& radius, float& sigma_spatial, float& sigma_color);

bool getBin(const std::string& binPath, int& width, int& height, uint8_t*& src1, myPixel*& src2);

float binDiff(const std::string& binPath_ref, const std::string& binPath_test);


//生成bin文件
template <typename T>
bool genBin(const std::string& outBinPath, const T* img, int width, int height){
    std::ofstream file(outBinPath, std::ios::binary);
    if(!file){
        std::cerr << "Error: Could not open file: " << outBinPath << std::endl;
        return false;
    }

    int channels = std::is_same_v<T, myPixel> ? 3 : 1;

    file.write(reinterpret_cast<char*>(&width), sizeof(int));
    file.write(reinterpret_cast<char*>(&height), sizeof(int));
    file.write(reinterpret_cast<char*>(&channels), sizeof(int));
    if constexpr(std::is_same_v<T, myPixel>){
        for(size_t i = 0; i < width * height; ++i){
            uint8_t rgb[3] = {img[i].R(), img[i].G(), img[i].B() };
            file.write(reinterpret_cast<char*>(rgb), 3);
        }
    }
    else{
        file.write(reinterpret_cast<const char*>(img), width * height);
    }

    file.close();
    //std::cout << "Successfully genBin: " << outBinPath << std::endl;
    return true;
}


//大图坐标相对小图坐标而言,取值范围在-radius ~ len + radius - 1
inline int mapReflect101(int p, int len){
    if(p < 0){//左越界
        return -p;
    }
    else if(p >= len){//右越界
        return 2 * (len - 1) - p;
    }
    else{//不越界
        return p;
    }
}

//将输入图像以reflect101模式进行边缘延拓
template <typename T>
T* Reflect101(const T* src, int width, int height, int radius){
    if(radius < 0 || radius >= width || radius >= height){
        std::cerr << "Error: unsupported radius" << std::endl;
        return nullptr;
    }

    int pixel_count_r101 = (width + 2 * radius) * (height + 2 * radius);
    T* src_r101 = new T[pixel_count_r101];

    for(int i = 0; i < height + 2 * radius; ++i){
        for(int j = 0; j < width + 2 * radius; ++j){
            int relative_i = i - radius;
            int relative_j = j - radius;
            int src_y = mapReflect101(relative_i, height);//纵坐标
            int src_x = mapReflect101(relative_j, width);//横坐标

            src_r101[i * (width + 2 * radius) + j] = src[src_y * width + src_x];
        }
    }
    return src_r101;
}