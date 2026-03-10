//包含和cv有关的函数, 如果服务器不包含opencv库则忽略该文件
#pragma once

#include <type_traits>
#include <cstring>
#include <cstdint>
#include <myPixel.h>

#ifdef HAS_CV

#ifdef PLATFORM_ILUVATAR
    #include <algorithm>
    namespace std {
        template<typename T1, typename T2>
        auto max(T1 a, T2 b) -> typename std::common_type<T1, T2>::type {
            return (a > b) ? a : b;
        }
    }
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wliteral-range"
#endif

#include <opencv2/opencv.hpp>

#ifdef PLATFORM_ILUVATAR
    #pragma clang diagnostic pop
#endif

bool genBin_cv(const std::string& outBinPath, const cv::Mat& img);

void genTester();

bool genImg_cv(const std::string& outImgPath, const cv::Mat& img, bool isRGB);

//cv版本bilateralFilter
template <typename T>
void runFilter_cv(const T* src, int radius, int width, int height, float sigma_spatial, float sigma_color, T* dst){
    if(src == nullptr){
        return;
    }

    int type = std::is_same_v<T, myPixel> ? CV_8UC4 : CV_8UC1;
    cv::Mat inputMat(height, width, type, const_cast<T*>(src));
    cv::Mat outputMat;
    
    if constexpr(std::is_same_v<T, myPixel>){
        cv::Mat tempBGR;
        cv::cvtColor(inputMat, tempBGR, cv::COLOR_RGBA2BGR); // 4 -> 3
        cv::Mat filteredBGR;
        cv::bilateralFilter(tempBGR, filteredBGR, 2 * radius + 1, sigma_color, sigma_spatial);
        cv::cvtColor(filteredBGR, outputMat, cv::COLOR_BGR2RGBA); // 3 -> 4
    }
    else{
        cv::bilateralFilter(inputMat, outputMat, 2 * radius + 1, sigma_color, sigma_spatial);
    }
    
    size_t total_bytes = width * height * sizeof(T);
    std::memcpy(dst, outputMat.data, total_bytes);
}


//生成图片, T = uint8_t/myPixel
template <typename T>
bool genImg(const std::string& outImgPath, const T* img, int width, int height){
    if constexpr(std::is_same_v<T, myPixel>){
        cv::Mat imgRGB(height, width, CV_8UC3, (void*)img);
        cv::Mat imgBGR;
        cv::cvtColor(imgRGB, imgBGR, cv::COLOR_RGB2BGR);    
        if (!cv::imwrite(outImgPath, imgBGR)){
            return false;
        }
    } 
    else if constexpr(std::is_same_v<T, uint8_t>) {
        cv::Mat imgGray(height, width, CV_8UC1, (void*)img);
        if (!cv::imwrite(outImgPath, imgGray)){
            return false;
        }
    } 
    else{
        std::cerr << "Error: Unsupported type for genImg." << std::endl;
        return false;
    }
    
    std::cout << "Successfully genImg: " << outImgPath << std::endl;
    return true;
}

#endif
