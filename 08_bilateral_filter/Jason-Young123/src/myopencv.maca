#include <myopencv.h>
#include <auxiliary.h>

#ifdef HAS_CV

//基于cv::Mat生成bin文件
bool genBin_cv(const std::string& outBinPath, const cv::Mat& img){
    std::ofstream file(outBinPath, std::ios::binary);
    if(!file){
        std::cerr << "Error: Could not open file: " << outBinPath << std::endl;
        return false;
    }
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    file.write(reinterpret_cast<char*>(&width), sizeof(int));
    file.write(reinterpret_cast<char*>(&height), sizeof(int));
    file.write(reinterpret_cast<char*>(&channels), sizeof(int));
    file.write(reinterpret_cast<char*>(img.data), size_t(width * height * channels));

    file.close();
    std::cout << "Successfully genBin: " << outBinPath << std::endl;
    return true;
}


//生成测试用例
void genTester(){
    std::string baseResource = "resource";
    std::string baseTester = "tester";

    struct Config {
        std::string subType;
        bool isRGB;
    };

    Config configs[] = {
        {"gray", false},
        {"rgb", true}
    };

    for (const auto& cfg : configs) {
        std::filesystem::path srcRoot = std::filesystem::path(baseResource) / cfg.subType;
        
        if (!std::filesystem::exists(srcRoot)) {
            std::cout << "Directory not found: " << srcRoot.string() << std::endl;
            continue;
        }

        //遍历: gray/4K, gray/others, rgb/4K, rgb/others
        for (const auto& entry : std::filesystem::recursive_directory_iterator(srcRoot)) {
            if (!entry.is_regular_file()) {
                continue;
            }

            std::filesystem::path filePath = entry.path();
            
            cv::Mat img = cv::imread(filePath.string(), cfg.isRGB ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
            if (img.empty()) continue;

            if (cfg.isRGB) {
                cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            }

            std::filesystem::path relativePath = std::filesystem::relative(filePath, srcRoot);
            std::filesystem::path outPath = std::filesystem::path(baseTester) / cfg.subType / relativePath;
            outPath.replace_extension(".bin");//.bin raw文件

            std::filesystem::create_directories(outPath.parent_path());

            if (genBin_cv(outPath.string(), img)) {
                std::cout << "[Tester] Converted: " << cfg.subType << "/" << relativePath.string() 
                          << " -> " << outPath.string() << std::endl;
            }
        }
    }
}


//基于cv::Mat生成图片
bool genImg_cv(const std::string& outImgPath, const cv::Mat& img, bool isRGB){
    cv::Mat finalImg;
    if(img.channels() == 3 && isRGB){
        cv::cvtColor(img, finalImg, cv::COLOR_RGB2BGR);
    }
    else{
        finalImg = img;
    }
    cv::imwrite(outImgPath, finalImg);
    
    std::cout << "Successfully genImg: " << outImgPath << std::endl;
    return true;
}


#endif