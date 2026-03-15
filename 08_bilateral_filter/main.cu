#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

// 错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr int blockSizeX = 32;
constexpr int blockSizeY = 8;
constexpr int maxGrayDistance = 255;
constexpr int maxColorDistance = 255 * 3;

struct BilateralParams {
    int radius = 5;
    float sigmaSpatial = 3.0f;
    float sigmaColor = 30.0f;
    bool useAdaptiveRadius = false;
    float noiseLevel = 0.0f;
};

// 灰度图双边滤波核函数
__global__ void BilateralFilterGrayKernel(
    const uchar* inputImage,
    uchar* outputImage,
    const float* spatialLookupWeights,
    const float* colorLookupWeights,
    int imageWidth,
    int imageHeight,
    size_t inputStep,
    size_t outputStep,
    int filterRadius
) {
    extern __shared__ uchar sharedTile[];

    int tileWidth = blockDim.x + 2 * filterRadius;
    int tileHeight = blockDim.y + 2 * filterRadius;
    int paddedWidth = imageWidth + 2 * filterRadius;
    int paddedHeight = imageHeight + 2 * filterRadius;

    for (int tileBlockY = blockIdx.y; tileBlockY * blockDim.y < imageHeight; tileBlockY += gridDim.y) {
        for (int tileBlockX = blockIdx.x; tileBlockX * blockDim.x < imageWidth; tileBlockX += gridDim.x) {
            int tileStartX = tileBlockX * blockDim.x;
            int tileStartY = tileBlockY * blockDim.y;

            for (int localY = threadIdx.y; localY < tileHeight; localY += blockDim.y) {
                for (int localX = threadIdx.x; localX < tileWidth; localX += blockDim.x) {
                    int globalX = tileStartX + localX;
                    int globalY = tileStartY + localY;

                    if (globalX > paddedWidth - 1) globalX = paddedWidth - 1;
                    if (globalY > paddedHeight - 1) globalY = paddedHeight - 1;

                    sharedTile[localY * tileWidth + localX] = inputImage[globalY * inputStep + globalX];
                }
            }
            __syncthreads();

            int pixelX = tileStartX + threadIdx.x;
            int pixelY = tileStartY + threadIdx.y;

            if (pixelX < imageWidth && pixelY < imageHeight) {
                int centerLocalX = threadIdx.x + filterRadius;
                int centerLocalY = threadIdx.y + filterRadius;

                float centerValue = static_cast<float>(sharedTile[centerLocalY * tileWidth + centerLocalX]);

                float weightValueSum = 0.0f;
                float weightSum = 0.0f;
                int spatialIndex = 0;

                for (int offsetY = -filterRadius; offsetY <= filterRadius; ++offsetY) {
                    for (int offsetX = -filterRadius; offsetX <= filterRadius; ++offsetX, ++spatialIndex) {
                        float nearValue = static_cast<float>(sharedTile[(centerLocalY + offsetY) * tileWidth + (centerLocalX + offsetX)]);
                        int colorDistance = static_cast<int>(fabsf(nearValue - centerValue));

                        if (colorDistance > maxGrayDistance) colorDistance = maxGrayDistance;

                        float finalWeight = spatialLookupWeights[spatialIndex] * colorLookupWeights[colorDistance];
                        weightValueSum += finalWeight * nearValue;
                        weightSum += finalWeight;
                    }
                }

                outputImage[pixelY * outputStep + pixelX] = static_cast<uchar>(weightValueSum / weightSum);
            }
            __syncthreads();
        }
    }
}

// 彩色图双边滤波核函数
__global__ void BilateralFilterColorKernel(
    const uchar* inputImage,
    uchar* outputImage,
    const float* spatialLookupWeights,
    const float* colorLookupWeights,
    int imageWidth,
    int imageHeight,
    size_t inputStep,
    size_t outputStep,
    int filterRadius
) {
    extern __shared__ uchar sharedTile[];

    int tileWidth = blockDim.x + 2 * filterRadius;
    int tileHeight = blockDim.y + 2 * filterRadius;
    int paddedWidth = imageWidth + 2 * filterRadius;
    int paddedHeight = imageHeight + 2 * filterRadius;

    // 网格跨步循环
    for (int tileBlockY = blockIdx.y; tileBlockY * blockDim.y < imageHeight; tileBlockY += gridDim.y) {
        for (int tileBlockX = blockIdx.x; tileBlockX * blockDim.x < imageWidth; tileBlockX += gridDim.x) {
            int tileStartX = tileBlockX * blockDim.x;
            int tileStartY = tileBlockY * blockDim.y;

            for (int localY = threadIdx.y; localY < tileHeight; localY += blockDim.y) {
                for (int localX = threadIdx.x; localX < tileWidth; localX += blockDim.x) {
                    int globalX = tileStartX + localX;
                    int globalY = tileStartY + localY;

                    if (globalX > paddedWidth - 1) globalX = paddedWidth - 1;
                    if (globalY > paddedHeight - 1) globalY = paddedHeight - 1;

                    size_t globalIndex = globalY * inputStep + globalX * 3;
                    size_t sharedIndex = (localY * tileWidth + localX) * 3;

                    sharedTile[sharedIndex + 0] = inputImage[globalIndex + 0];
                    sharedTile[sharedIndex + 1] = inputImage[globalIndex + 1];
                    sharedTile[sharedIndex + 2] = inputImage[globalIndex + 2];
                }
            }
            __syncthreads();

            int pixelX = tileStartX + threadIdx.x;
            int pixelY = tileStartY + threadIdx.y;

            if (pixelX < imageWidth && pixelY < imageHeight) {
                int centerLocalX = threadIdx.x + filterRadius;
                int centerLocalY = threadIdx.y + filterRadius;

                size_t centerIndex = (centerLocalY * tileWidth + centerLocalX) * 3;

                float centerBlue = static_cast<float>(sharedTile[centerIndex + 0]);
                float centerGreen = static_cast<float>(sharedTile[centerIndex + 1]);
                float centerRed = static_cast<float>(sharedTile[centerIndex + 2]);

                float blueSum = 0.0f;
                float greenSum = 0.0f;
                float redSum = 0.0f;
                float weightSum = 0.0f;
                int spatialIndex = 0;

                for (int offsetY = -filterRadius; offsetY <= filterRadius; ++offsetY) {
                    for (int offsetX = -filterRadius; offsetX <= filterRadius; ++offsetX, ++spatialIndex) {
                        size_t nearIndex = ((centerLocalY + offsetY) * tileWidth + (centerLocalX + offsetX)) * 3;
                        
						float nearBlue = static_cast<float>(sharedTile[nearIndex + 0]);
                        float nearGreen = static_cast<float>(sharedTile[nearIndex + 1]);
                        float nearRed = static_cast<float>(sharedTile[nearIndex + 2]);

                        int colorDistance =
                            static_cast<int>(fabsf(nearBlue - centerBlue)) +
                            static_cast<int>(fabsf(nearGreen - centerGreen)) +
                            static_cast<int>(fabsf(nearRed - centerRed));

                        if (colorDistance > maxColorDistance) colorDistance = maxColorDistance;
                        float finalWeight = spatialLookupWeights[spatialIndex] * colorLookupWeights[colorDistance];
                        
						blueSum += finalWeight * nearBlue;
                        greenSum += finalWeight * nearGreen;
                        redSum += finalWeight * nearRed;
                        weightSum += finalWeight;
                    }
                }

                size_t outputIndex = pixelY * outputStep + pixelX * 3;

                outputImage[outputIndex + 0] = static_cast<uchar>(blueSum / weightSum);
                outputImage[outputIndex + 1] = static_cast<uchar>(greenSum / weightSum);
                outputImage[outputIndex + 2] = static_cast<uchar>(redSum / weightSum);
            }
            __syncthreads();
        }
    }
}

// 创建空间权重和颜色权重查找表
void lookupWeights(
    std::vector<float>& spatialLookupWeights,
    std::vector<float>& colorLookupWeights,
    int filterRadius,
    float spatialSigma,
    float colorSigma,
    int colorTableSize
) {
    int windowLength = 2 * filterRadius + 1;

    spatialLookupWeights.resize(windowLength * windowLength);
    colorLookupWeights.resize(colorTableSize + 1);

    float spatialFactor = -0.5f / (spatialSigma * spatialSigma);
    float colorCoefficient = -0.5f / (colorSigma * colorSigma);
    int spatialIndex = 0;

    for (int offsetY = -filterRadius; offsetY <= filterRadius; ++offsetY) {
        for (int offsetX = -filterRadius; offsetX <= filterRadius; ++offsetX, ++spatialIndex) {
            float distanceSquare = static_cast<float>(offsetX * offsetX + offsetY * offsetY);
            spatialLookupWeights[spatialIndex] = expf(distanceSquare * spatialFactor);
        }
    }

    for (int colorDistance = 0; colorDistance <= colorTableSize; ++colorDistance) {
        colorLookupWeights[colorDistance] = expf(colorDistance * colorDistance * colorCoefficient);
    }
}

float calculateMeanAbsoluteError(const cv::Mat& image1, const cv::Mat& image2) {
    cv::Mat diff;
    cv::absdiff(image1, image2, diff);
    cv::Scalar meanVal = cv::mean(diff);
    float mae = 0.0f;
    for (int c = 0; c < image1.channels(); ++c) {
        mae += static_cast<float>(meanVal[c]);
    }
    return mae / image1.channels();
}

float estimateNoiseLevel(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_32F);
    cv::Mat absLaplacian = cv::abs(laplacian);

    cv::Scalar meanVal, stdVal;
    cv::meanStdDev(absLaplacian, meanVal, stdVal);
    return static_cast<float>(stdVal[0]);
}

int selectAdaptiveRadius(const cv::Mat& image, float baseRadius, float noiseLevel) {
    if (noiseLevel <= 0) {
        noiseLevel = estimateNoiseLevel(image);
    }

    float noiseFactor = std::min(noiseLevel / 30.0f, 2.0f);
    int adaptiveRadius = static_cast<int>(baseRadius * (1.0f + noiseFactor * 0.5f));
    return std::max(2, std::min(adaptiveRadius, 10));
}

bool parseParamsFile(const std::string& filename, BilateralParams& params) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Cannot open params file: " << filename << ", using default values." << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string key;
        if (std::getline(iss, key, '=')) {
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);

            std::string value;
            if (std::getline(iss, value, '#')) {
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
            }

            if (key == "radius") {
                params.radius = std::stoi(value);
            } else if (key == "sigma_spatial") {
                params.sigmaSpatial = std::stof(value);
            } else if (key == "sigma_color") {
                params.sigmaColor = std::stof(value);
            } else if (key == "use_adaptive_radius") {
                params.useAdaptiveRadius = (value == "true" || value == "1");
            }
        }
    }
    return true;
}

bool writeRawImage(const std::string& filename, const cv::Mat& image) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot write to file: " << filename << std::endl;
        return false;
    }

    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();

    file.write(reinterpret_cast<const char*>(&width), sizeof(int));
    file.write(reinterpret_cast<const char*>(&height), sizeof(int));
    file.write(reinterpret_cast<const char*>(&channels), sizeof(int));

    for (int y = 0; y < height; ++y) {
        file.write(reinterpret_cast<const char*>(image.ptr(y)), width * channels);
    }

    return file.good();
}

cv::Mat readRawImage(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << filename << std::endl;
        return cv::Mat();
    }

    int width, height, channels;
    file.read(reinterpret_cast<char*>(&width), sizeof(int));
    file.read(reinterpret_cast<char*>(&height), sizeof(int));
    file.read(reinterpret_cast<char*>(&channels), sizeof(int));

    int type = (channels == 1) ? CV_8UC1 : CV_8UC3;
    cv::Mat image(height, width, type);

    for (int y = 0; y < height; ++y) {
        file.read(reinterpret_cast<char*>(image.ptr(y)), width * channels);
    }

    return image;
}

void writePerformanceLog(
    const std::string& filename,
    const std::string& imagePath,
    int width,
    int height,
    int channels,
    int radius,
    float spatialSigma,
    float colorSigma,
    double gpuTimeMs,
    double cpuTimeMs,
    float mae
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot write log file: " << filename << std::endl;
        return;
    }

    double totalPixels = static_cast<double>(width) * height;
    double gpuThroughput = totalPixels / (gpuTimeMs / 1000.0) / 1000000.0;
    double cpuThroughput = totalPixels / (cpuTimeMs / 1000.0) / 1000000.0;
    double speedup = cpuTimeMs / gpuTimeMs;

    file << "========== Bilateral Filter Performance Log ==========" << std::endl;
    file << "Image: " << imagePath << std::endl;
    file << "Resolution: " << width << "x" << height << std::endl;
    file << "Channels: " << channels << std::endl;
    file << "Total Pixels: " << totalPixels << std::endl;
    file << std::endl;
    file << "Filter Parameters:" << std::endl;
    file << "  Radius: " << radius << std::endl;
    file << "  Sigma Spatial: " << spatialSigma << std::endl;
    file << "  Sigma Color: " << colorSigma << std::endl;
    file << std::endl;
    file << "Performance Metrics:" << std::endl;
    file << "  GPU Time: " << gpuTimeMs << " ms" << std::endl;
    file << "  GPU Throughput: " << gpuThroughput << " MP/s" << std::endl;
    file << "  CPU Time: " << cpuTimeMs << " ms" << std::endl;
    file << "  CPU Throughput: " << cpuThroughput << " MP/s" << std::endl;
    file << "  Speedup (CPU/GPU): " << speedup << "x" << std::endl;
    file << std::endl;
    file << "Correctness:" << std::endl;
    file << "  MAE vs OpenCV: " << mae << std::endl;
    file << "  Pass (MAE < 1.0): " << (mae < 1.0 ? "YES" : "NO") << std::endl;
    file << "======================================================" << std::endl;
}

double measureGPUTime(
    const cv::Mat& inputImage,
    const cv::Mat& paddedImage,
    cv::Mat& resultImage,
    const std::vector<float>& spatialLookupWeights,
    const std::vector<float>& colorLookupWeights,
    int filterRadius
) {
    uchar* deviceInputImage = nullptr;
    uchar* deviceOutputImage = nullptr;
    float* deviceSpatialLookupWeights = nullptr;
    float* deviceColorLookupWeights = nullptr;

    CUDA_CHECK(cudaMalloc(&deviceInputImage, paddedImage.step * paddedImage.rows));
    CUDA_CHECK(cudaMalloc(&deviceOutputImage, resultImage.step * resultImage.rows));
    CUDA_CHECK(cudaMalloc(&deviceSpatialLookupWeights, spatialLookupWeights.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&deviceColorLookupWeights, colorLookupWeights.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(deviceInputImage, paddedImage.data, paddedImage.step * paddedImage.rows, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceSpatialLookupWeights, spatialLookupWeights.data(), spatialLookupWeights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceColorLookupWeights, colorLookupWeights.data(), colorLookupWeights.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(blockSizeX, blockSizeY);

    //如果 block 数量直接铺满整张图，网格跨步循环往往只会执行一轮；适当限制 grid 后，一个 block 才会继续跨步处理后续 tile。
    cudaDeviceProp deviceInfo{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceInfo, 0));
    int neededGridX = (inputImage.cols + blockSizeX - 1) / blockSizeX;
    int neededGridY = (inputImage.rows + blockSizeY - 1) / blockSizeY;
    int launchGridX = std::max(1, std::min(neededGridX, deviceInfo.multiProcessorCount / 4));
    int launchGridY = std::max(1, std::min(neededGridY, 4));
    dim3 blockGroup(launchGridX, launchGridY);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    //共享内存 tile 的大小取决于线程块大小、滤波半径和图像通道数
    if (inputImage.channels() == 1) {
        size_t sharedMemoryBytes =
            static_cast<size_t>(blockSize.x + 2 * filterRadius) *
            static_cast<size_t>(blockSize.y + 2 * filterRadius) *
            sizeof(uchar);

        BilateralFilterGrayKernel<<<blockGroup, blockSize, sharedMemoryBytes>>>(
            deviceInputImage, deviceOutputImage, deviceSpatialLookupWeights, deviceColorLookupWeights,
            inputImage.cols, inputImage.rows, paddedImage.step, resultImage.step, filterRadius);
    } else {
        size_t sharedMemoryBytes =
            static_cast<size_t>(blockSize.x + 2 * filterRadius) *
            static_cast<size_t>(blockSize.y + 2 * filterRadius) *
            3 * sizeof(uchar);

        BilateralFilterColorKernel<<<blockGroup, blockSize, sharedMemoryBytes>>>(
            deviceInputImage, deviceOutputImage, deviceSpatialLookupWeights, deviceColorLookupWeights,
            inputImage.cols, inputImage.rows, paddedImage.step, resultImage.step, filterRadius);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iterations = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        if (inputImage.channels() == 1) {
            size_t sharedMemoryBytes =
                static_cast<size_t>(blockSize.x + 2 * filterRadius) *
                static_cast<size_t>(blockSize.y + 2 * filterRadius) *
                sizeof(uchar);
            BilateralFilterGrayKernel<<<blockGroup, blockSize, sharedMemoryBytes>>>(
                deviceInputImage, deviceOutputImage, deviceSpatialLookupWeights, deviceColorLookupWeights,
                inputImage.cols, inputImage.rows, paddedImage.step, resultImage.step, filterRadius);
        } else {
            size_t sharedMemoryBytes =
                static_cast<size_t>(blockSize.x + 2 * filterRadius) *
                static_cast<size_t>(blockSize.y + 2 * filterRadius) *
                3 * sizeof(uchar);
            BilateralFilterColorKernel<<<blockGroup, blockSize, sharedMemoryBytes>>>(
                deviceInputImage, deviceOutputImage, deviceSpatialLookupWeights, deviceColorLookupWeights,
                inputImage.cols, inputImage.rows, paddedImage.step, resultImage.step, filterRadius);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    double avgTime = elapsedTime / iterations;

    CUDA_CHECK(cudaMemcpy(resultImage.data, deviceOutputImage, resultImage.step * resultImage.rows, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(deviceInputImage));
    CUDA_CHECK(cudaFree(deviceOutputImage));
    CUDA_CHECK(cudaFree(deviceSpatialLookupWeights));
    CUDA_CHECK(cudaFree(deviceColorLookupWeights));

    return avgTime;
}

double measureCPUTime(const cv::Mat& inputImage, cv::Mat& outputImage, int diameter, float colorSigma, float spatialSigma) {
    const int iterations = 5;
    cv::bilateralFilter(inputImage, outputImage, diameter, colorSigma, spatialSigma);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        cv::bilateralFilter(inputImage, outputImage, diameter, colorSigma, spatialSigma);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / iterations;
}

int main(int argc, char** argv) {
    std::string imagePath = (argc > 1) ? argv[1] : "test.jpg";
    std::string paramsFile = (argc > 2) ? argv[2] : "params.txt";

    BilateralParams params;
    parseParamsFile(paramsFile, params);

    std::cout << "Parameters:" << std::endl;
    std::cout << "  Radius: " << params.radius << std::endl;
    std::cout << "  Sigma Spatial: " << params.sigmaSpatial << std::endl;
    std::cout << "  Sigma Color: " << params.sigmaColor << std::endl;
    std::cout << "  Use Adaptive Radius: " << (params.useAdaptiveRadius ? "true" : "false") << std::endl;

    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    if (inputImage.empty()) {
        std::cerr << "Error: Cannot load image: " << imagePath << std::endl;
        return EXIT_FAILURE;
    }

    if (inputImage.channels() == 4) {
        cv::cvtColor(inputImage, inputImage, cv::COLOR_BGRA2BGR);
    }

    if (inputImage.channels() != 1 && inputImage.channels() != 3) {
        std::cerr << "Error: Only 1-channel (grayscale) or 3-channel (RGB) images are supported." << std::endl;
        return EXIT_FAILURE;
    }

    int filterRadius = params.radius;
    if (params.useAdaptiveRadius) {
        filterRadius = selectAdaptiveRadius(inputImage, params.radius, params.noiseLevel);
        std::cout << "Adaptive radius selected: " << filterRadius << std::endl;
    }

    int colorTableSize = (inputImage.channels() == 1) ? maxGrayDistance : maxColorDistance;
    std::vector<float> spatialLookupWeights;
    std::vector<float> colorLookupWeights;
    lookupWeights(spatialLookupWeights, colorLookupWeights, filterRadius, params.sigmaSpatial, params.sigmaColor, colorTableSize);

    cv::Mat paddedImage;
    cv::copyMakeBorder(inputImage, paddedImage, filterRadius, filterRadius, filterRadius, filterRadius, cv::BORDER_REFLECT);
    cv::Mat resultImage = cv::Mat::zeros(inputImage.size(), inputImage.type());

    double gpuTimeMs = measureGPUTime(inputImage, paddedImage, resultImage, spatialLookupWeights, colorLookupWeights, filterRadius);

    cv::Mat referenceImage;
    double cpuTimeMs = measureCPUTime(inputImage, referenceImage, 2 * filterRadius + 1, params.sigmaColor, params.sigmaSpatial);

    float mae = calculateMeanAbsoluteError(resultImage, referenceImage);

    std::cout << std::endl;
    std::cout << "========== Results ==========" << std::endl;
    std::cout << "Image: " << imagePath << " (" << inputImage.cols << "x" << inputImage.rows << ", " << inputImage.channels() << " channels)" << std::endl;
    std::cout << "Filter Radius: " << filterRadius << std::endl;
    std::cout << std::endl;
    std::cout << "Performance:" << std::endl;
    std::cout << "  GPU Time: " << gpuTimeMs << " ms" << std::endl;
    std::cout << "  GPU Throughput: " << (inputImage.cols * inputImage.rows / (gpuTimeMs / 1000.0) / 1000000.0) << " MP/s" << std::endl;
    std::cout << "  CPU Time: " << cpuTimeMs << " ms" << std::endl;
    std::cout << "  CPU Throughput: " << (inputImage.cols * inputImage.rows / (cpuTimeMs / 1000.0) / 1000000.0) << " MP/s" << std::endl;
    std::cout << "  Speedup (CPU/GPU): " << (cpuTimeMs / gpuTimeMs) << "x" << std::endl;
    std::cout << std::endl;
    std::cout << "Correctness:" << std::endl;
    std::cout << "  MAE vs OpenCV: " << mae << std::endl;
    std::cout << "  Pass (MAE < 1.0): " << (mae < 1.0 ? "YES" : "NO") << std::endl;
    std::cout << "=============================" << std::endl;

    std::string outputFilename = "bilateralFilter_result.raw";
    if (!writeRawImage(outputFilename, resultImage)) {
        std::cerr << "Warning: Failed to write output raw file." << std::endl;
    }

    cv::imwrite("bilateralFilter_result.jpg", resultImage);

    writePerformanceLog(
        "performance_log.txt",
        imagePath,
        inputImage.cols,
        inputImage.rows,
        inputImage.channels(),
        filterRadius,
        params.sigmaSpatial,
        params.sigmaColor,
        gpuTimeMs,
        cpuTimeMs,
        mae
    );

    if (mae >= 1.0f) {
        std::cerr << "Warning: MAE (" << mae << ") >= 1.0, results may not meet accuracy requirements." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
