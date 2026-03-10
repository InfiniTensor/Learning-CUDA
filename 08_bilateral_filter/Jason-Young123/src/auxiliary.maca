#include <auxiliary.h>

//获取cfg文件
bool getCfg(const std::string& cfgPath, int& radius, float& sigma_spatial, float& sigma_color){
    std::ifstream file(cfgPath);
    if(!file){
        std::cerr << "Error: Could not open file: " << cfgPath << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("radius") != std::string::npos) 
            sscanf(line.c_str(), " radius = %d", &radius);
        else if (line.find("sigma_spatial") != std::string::npos)
            sscanf(line.c_str(), " sigma_spatial = %f", &sigma_spatial);
        else if (line.find("sigma_color") != std::string::npos)
            sscanf(line.c_str(), " sigma_color = %f", &sigma_color);
    }
    file.close();
    return true;
}


//获取bin文件
bool getBin(const std::string& binPath, int& width, int& height, uint8_t*& src1, myPixel*& src2){
    std::ifstream file(binPath, std::ios::binary);
    if(!file){
        std::cerr << "Error: Could not open file: " << binPath << std::endl;
        return false;
    }

    //step1: 获取头信息
    int channels;
    file.read(reinterpret_cast<char*>(&width), sizeof(int));
    file.read(reinterpret_cast<char*>(&height), sizeof(int));
    file.read(reinterpret_cast<char*>(&channels), sizeof(int));

    //step2: 获取像素信息
    size_t pixel_count = static_cast<size_t>(width) * height;
    if(channels == 3){
        src1 = nullptr;
        src2 = new myPixel[pixel_count];

        std::vector<uint8_t> tmp_buffer(pixel_count * 3);
        file.read(reinterpret_cast<char*>(tmp_buffer.data()), pixel_count * 3);
        for(size_t i = 0; i < pixel_count; ++i) {
            uint8_t r = tmp_buffer[i * 3 + 0];
            uint8_t g = tmp_buffer[i * 3 + 1];
            uint8_t b = tmp_buffer[i * 3 + 2];
            src2[i] = myPixel(r, g, b); // 构造函数会自动把 _A 设为 255
        }
    }
    else if(channels == 1){
        src1 = new uint8_t[width * height];
        src2 = nullptr;
        file.read(reinterpret_cast<char*>(src1), pixel_count);
    }
    else{
        std::cerr << "Error: Unsupported channels = " << channels << std::endl;
        return false;
    }
    
    file.close();
    return true;
}



//对比bin文件并返回MAE
float binDiff(const std::string& binPath_ref, const std::string& binPath_test){
    std::ifstream file_ref(binPath_ref, std::ios::binary);
    std::ifstream file_test(binPath_test, std::ios::binary);

    if(!file_ref || !file_test){
        std::cerr << "Error: Could not open file for comparison." << std::endl;
        return -1.0f;
    }

    int w1, h1, c1;
    int w2, h2, c2;
    file_ref.read(reinterpret_cast<char*>(&w1), sizeof(int));
    file_ref.read(reinterpret_cast<char*>(&h1), sizeof(int));
    file_ref.read(reinterpret_cast<char*>(&c1), sizeof(int));
    file_test.read(reinterpret_cast<char*>(&w2), sizeof(int));
    file_test.read(reinterpret_cast<char*>(&h2), sizeof(int));
    file_test.read(reinterpret_cast<char*>(&c2), sizeof(int));
    if(w1 != w2 || h1 != h2 || c1 != c2){
        std::cerr << "Error: Image size not match." << std::endl;
        return -1.0f;
    }

    size_t totalPixels = static_cast<size_t>(w1) * h1 * c1;
    std::vector<unsigned char> data_ref(totalPixels);
    std::vector<unsigned char> data_test(totalPixels);
    file_ref.read(reinterpret_cast<char*>(data_ref.data()), totalPixels);
    file_test.read(reinterpret_cast<char*>(data_test.data()), totalPixels);

    file_ref.close();
    file_test.close();

    long totalError = 0;
    for (size_t i = 0; i < totalPixels; ++i) {
        totalError += std::abs(int(data_ref[i]) - int(data_test[i]));
    }

    float MAE = double(totalError) / double(totalPixels);
    return MAE;
}
