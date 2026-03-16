#include "params.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

// 解析 params.txt 文件
FilterParams load_params(const std::string& filepath) {
    // 设置默认值
    FilterParams params;
    params.radius = 5;
    params.sigma_spatial = 3.0f;
    params.sigma_color = 30.0f;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "警告：无法打开参数文件 " << filepath
                  << "，使用默认参数" << std::endl;
        return params;
    }

    std::string line;
    while (std::getline(file, line)) {
        // 去掉 # 后面的注释部分
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        // 跳过空行
        if (line.find('=') == std::string::npos) {
            continue;
        }

        // 按 '=' 分割成 key 和 value
        std::istringstream iss(line);
        std::string key;
        std::getline(iss, key, '=');
        std::string value;
        std::getline(iss, value);

        // 去掉首尾空格
        auto trim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(" \t"));
            s.erase(s.find_last_not_of(" \t") + 1);
        };
        trim(key);
        trim(value);

        // 根据 key 名称赋值给对应字段
        if (key == "radius") {
            params.radius = std::stoi(value);
        } else if (key == "sigma_spatial") {
            params.sigma_spatial = std::stof(value);
        } else if (key == "sigma_color") {
            params.sigma_color = std::stof(value);
        }
    }

    return params;
}
