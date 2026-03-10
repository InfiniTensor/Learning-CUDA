#include <bilateral.h>
#include <auxiliary.h>
#include <myopencv.h>




//掐头去尾求平均用时
float ave_time(const std::vector<float>& time_ms, size_t size) {
    if (size < 3 || time_ms.size() < size) {
        float sum = std::accumulate(time_ms.begin(), time_ms.begin() + std::min(size, time_ms.size()), 0.0f);
        return sum / std::min(size, time_ms.size());
    }

    auto it_begin = time_ms.begin();
    auto it_end = time_ms.begin() + size;

    //遍历寻找最大最小值
    auto [min_it, max_it] = std::minmax_element(it_begin, it_end);
    float min_val = *min_it;
    float max_val = *max_it;

    float sum = std::accumulate(it_begin, it_end, 0.0f);

    return (sum - min_val - max_val) / (float)(size - 2);
}




int autoSelectRadius(const uint8_t* src_gray, const myPixel* src_rgb, int channels, int width, int height, int radius, float sigma_spatial){
    if(radius > 0){//指定合法半径,直接使用
        return radius;
    }

    float base_r = 3.0f * sigma_spatial;

    long long total_diff = 0;
    int samples = 0;
    int stride_x = std::max(1, width / 50);
    int stride_y = std::max(1, height / 50);

    for(int i = 0; i < height - 1; i += stride_y){
        for(int j = 0; j < width - 1; j += stride_x){
            int idx = i * width + j;
            int idx_right = idx + 1;
            int idx_down = std::min(i + 1, height - 1) * width + j;
            if(channels == 1 && src_gray){
                total_diff += std::abs(int(src_gray[idx]) - int(src_gray[idx_right]));
                total_diff += std::abs(int(src_gray[idx]) - int(src_gray[idx_down]));
            }
            else if(channels == 3 && src_rgb){
                total_diff += std::abs(int(src_rgb[idx].R()) - int(src_rgb[idx_right].R()));
                total_diff += std::abs(int(src_rgb[idx].G()) - int(src_rgb[idx_right].G()));
                total_diff += std::abs(int(src_rgb[idx].B()) - int(src_rgb[idx_right].B()));
                total_diff += std::abs(int(src_rgb[idx].R()) - int(src_rgb[idx_down].R()));
                total_diff += std::abs(int(src_rgb[idx].G()) - int(src_rgb[idx_down].G()));
                total_diff += std::abs(int(src_rgb[idx].B()) - int(src_rgb[idx_down].B()));
            }
            else{

            }
            samples++;
        }
    }

    double divisor = (channels == 1) ? 2.0 : 6.0;
    float avg_diff = (samples > 0) ? (double)total_diff / (samples * divisor) : 0;

    float factor = 0.6f + (avg_diff - 2.0f) * (1.4f - 0.6f) / (15.0f - 2.0f);
    factor = std::max(0.6f, std::min(1.4f, factor));

    int final_r = (int)(base_r * factor + 0.5f);
    if (final_r < 3) final_r = 3; 
    if (final_r > 10) final_r = 10; 

    return final_r;

}





void runSingleCase(const std::filesystem::path& casePath, int radius, float sigma_spatial, float sigma_color, int warmup_round, int test_round){
    std::string fileName = casePath.filename().string();//文件名,如cameraman.bin
    std::string filePath = casePath.string();//文件路径
    //std::string outPath = "result/gpu/" + fileName;
    //std::string refPath = "result/opencv/" + fileName;

    int width, height;
    uint8_t* src_gray = nullptr;
    myPixel* src_rgb = nullptr;
    getBin(filePath, width, height, src_gray, src_rgb);
    int channels = src_gray ? 1 : 3;
    float sigma_color_sq = 2 * sigma_color * sigma_color;
    float sigma_spatial_sq = 2 * sigma_spatial * sigma_spatial;


    std::string outPath = (channels == 1) ? ("result/gpu/gray/" + fileName) : ("result/gpu/rgb/" + fileName);
    std::string refPath = (channels == 1) ? ("result/opencv/gray/" + fileName) : ("result/opencv/rgb/" + fileName);


    std::cout << BLUE << " [ " << casePath.stem().string() << " ] " << ":";
    if (width == 3840 && height == 2160) {
        std::cout << " 4K(3840 * 2160 * " << channels << ")" << RESET << std::endl;
    } else {
        std::cout << " " << width << " * " << height << " * " << channels << RESET << std::endl;
    }
    int radius1 = autoSelectRadius(src_gray, src_rgb, channels, width, height, radius, sigma_spatial);//自适应调整半径
    std::string msg = (radius <= 0) ? "(Auto)" : "";
    std::cout << "Param:  " << "radius = " << radius1 << msg << "  sigma_s = " << sigma_spatial << "  sigma_c = " << sigma_color << std::endl;

    auto runCase = [&](auto* h_src){
        using Type = std::remove_pointer_t<decltype(h_src)>;
        size_t src_size = static_cast<size_t>(width + 2 * radius1) * (height + 2 * radius1) * sizeof(Type);
        size_t dst_size = static_cast<size_t>(width) * height * sizeof(Type);
        Type* h_src_r101 = nullptr, *h_dst = new Type[width * height];
        h_src_r101 = Reflect101<Type>(h_src, width, height, radius1);

        //gpu版本
        RUNTIME_CHECK(musaHostRegister(h_src_r101, src_size, musaHostRegisterDefault));
        RUNTIME_CHECK(musaHostRegister(h_dst, dst_size, musaHostRegisterDefault));

        //预分配显存, 只需进行一次
        Type *d_src, *d_dst;
        musaStream_t stream;
        RUNTIME_CHECK(musaMalloc(&d_src, src_size));
        RUNTIME_CHECK(musaMalloc(&d_dst, dst_size));
        RUNTIME_CHECK(musaStreamCreate(&stream));
        
        for(int i = 0; i < warmup_round; ++i){//warmup
            runFilterPure<Type>(h_src_r101, d_src, d_dst, h_dst, radius1, width, height, sigma_spatial_sq, sigma_color_sq, src_size, dst_size, stream);
        }
        std::vector<float> time_ms;
        for(int i = 0; i < test_round; ++i){//test
            auto start = std::chrono::high_resolution_clock::now();
            runFilterPure<Type>(h_src_r101, d_src, d_dst, h_dst, radius1, width, height, sigma_spatial_sq, sigma_color_sq, src_size, dst_size, stream);
            auto end = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration<float, std::milli>(end - start).count();
            time_ms.push_back(duration);
        }
        
        //清理资源, 只需进行一次
        RUNTIME_CHECK(musaHostUnregister(h_src_r101));
        RUNTIME_CHECK(musaHostUnregister(h_dst));
        RUNTIME_CHECK(musaFree(d_src));
        RUNTIME_CHECK(musaFree(d_dst));
        RUNTIME_CHECK(musaStreamDestroy(stream));

        float time_ms_gpu = ave_time(time_ms, test_round);
        float throughput_gpu = (width * height / 1000000.0f) / (time_ms_gpu / 1000.0f);
        genBin<Type>(outPath, h_dst, width, height);
        printf("gpu( musa ): Time: %8.3f ms | Throughput: %8.2f MP/s \n", time_ms_gpu, throughput_gpu);

#ifdef HAS_CV
        std::vector<float> time_ms1;
        for(int i = 0; i < test_round; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            runFilter_cv<Type>(h_src, radius1, width, height, sigma_spatial, sigma_color, h_dst);
            auto end = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration<float, std::milli>(end - start).count();
            time_ms1.push_back(duration);
        }
        float time_ms_cv = ave_time(time_ms1, test_round);
        float throughput_cv = (width * height / 1000000.0f) / (time_ms_cv / 1000.0f);
        genBin<Type>(refPath, h_dst, width, height);

        printf("cpu(opencv): Time: %8.3f ms | Throughput: %8.2f MP/s \n", time_ms_cv, throughput_cv);
        
        std::cout << "Acceleration Ratio: " << throughput_gpu / throughput_cv << std::endl;
        
        float MAE = binDiff(refPath, outPath);
        if (MAE >= 1) {
            std::cout << "MAE: " << RED << MAE << " ( failed )" << RESET << std::endl << std::endl;
        } else {
            std::cout << "MAE: " << GREEN << MAE << " ( passed )" << RESET << std::endl << std::endl;
        }
#endif

        delete[] h_src_r101; delete[] h_dst;
    };

    if (channels == 1) {
        runCase(src_gray);
        delete[] src_gray;
    } else {
        runCase(src_rgb);
        delete[] src_rgb;
    }

}





void runAll(const std::string testerPath, int warmup_round, int test_round){
    //step0: preparation, get configuration info
    std::string cfgPath = testerPath + "/config.txt";//tester/config.txt
    int radius = 5;
    float sigma_color = 30.0f;
    float sigma_spatial = 3.0f;
    getCfg(cfgPath, radius, sigma_spatial, sigma_color);
    if(sigma_color <= 0.0f || sigma_spatial <= 0.0f){
        std::cout << "Invalid argument: sigma_spatial/sigma_color, Quit..." << std::endl;
        return;
    }

    float sigma_color_sq = 2 * sigma_color * sigma_color;
    float sigma_spatial_sq = 2 * sigma_spatial * sigma_spatial;
    int ret = std::system("mkdir -p result/gpu/gray result/gpu/rgb result/opencv/gray result/opencv/rgb");//存储结果
    (void)ret;

    //init spatial_lut
    //float h_spatial_lut_data[6][6];
    //for(int i = 0; i < 6; ++i){
    //    for(int j = 0; j < 6; ++j){
    //        h_spatial_lut_data[i][j] = expf(-(float)(i * i + j * j) / (2.0f * sigma_spatial * sigma_spatial));
    //    }
    //}
    //musaMemcpyToSymbol(spatial_lut_data, h_spatial_lut_data, sizeof(h_spatial_lut_data));
    
    //init color_lut
    //float h_color_lut_data[768];
    //for(int i = 0; i < 768; ++i){
    //    h_color_lut_data[i] = expf(-(float)(i * i) / (2.0f * sigma_color * sigma_color));
    //}
    //musaMemcpyToSymbol(color_lut_data, h_color_lut_data, sizeof(h_color_lut_data));


    //step1: 灰度测试
    std::cout << "\n" << BOLD << BLUE << " ### Tester of gray images: ### " << RESET << std::endl << std::endl;
    std::string grayPath = testerPath + "/gray/4K";
    for(const auto& entry : std::filesystem::recursive_directory_iterator(grayPath)){
        if(entry.is_regular_file()){
            runSingleCase(entry.path(), radius, sigma_spatial, sigma_color, warmup_round, test_round);
        }
    }


    //step2: RGB测试
    std::cout << "\n" << BOLD << BLUE << " ### Tester of RGB images: ### " << RESET << std::endl << std::endl;
    std::string rgbPath = testerPath + "/rgb/4K";
    for(const auto& entry : std::filesystem::recursive_directory_iterator(rgbPath)){
        if(entry.is_regular_file()){
            runSingleCase(entry.path(), radius, sigma_spatial, sigma_color, warmup_round, test_round);
        }
    }

}











