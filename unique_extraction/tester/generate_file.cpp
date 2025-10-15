#include <omp.h>
#include <iostream>
#include <random>
#include <cstdint>

#define FILE_DATA_SIZE (1 << 24)

int main(int argc, char* argv[]) {
    
    // uint64_t data_size = 1 << 27;
    uint64_t data_size = 1 << 20;

    if (argc > 2) {
        std::cerr << "参数错误\n";
        return 1;
    } else if (argc == 2) {
        bool ok = true;
        uint64_t tmp = 0, size = 0;
        for (int i = 0; argv[1][i] != '\0'; i++) {
            if ('0' <= argv[1][i] && argv[1][i] <= '9')
                tmp = tmp * 10 + (argv[1][i] - '0');
            else {
                ok = false;
                std::cerr << "请输入正整数\n";
            }
            size ++;
        }
        if (!ok)    return 1;
        if (size >= 10) {
            std::cerr << "太大了！！！\n";
            return 1;
        }
        data_size = tmp;
    }

    printf("data_size: %llu\n", data_size);

    int n = (data_size + FILE_DATA_SIZE - 1) / FILE_DATA_SIZE;

    const int num_threads = omp_get_max_threads();

    std::vector<uint64_t> seed(num_threads);
    
    {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        for (int i = 0; i < num_threads; i++)
            seed[i] = gen();
    }

    #pragma omp parallel for 
    for (int i = 0; i < n; i++) {

        int tid = omp_get_thread_num();
        int data_len = std::min((uint64_t)FILE_DATA_SIZE, data_size - i * FILE_DATA_SIZE);
        char path[100];
        snprintf(path, 100 * sizeof(char), "./tester/input/uint64-%llu-%llu.in", data_len, i);
        
        FILE* fp;

        fp = std::fopen(path, "w");
    
        if (!fp) {
            fprintf(stderr, "无法创建文件 %s\n", path);
            continue;
        } else {
            fprintf(stdout, "创建成功 %s\n", path);
        }

        std::mt19937_64 gen(seed[tid]);
        std::uniform_int_distribution<uint64_t> dis;
    
        for (uint64_t j = 0; j < data_len; j ++)
            fprintf(fp, "%llu ", dis(gen) & 0xFFFFFFFF00000000);
    
        fclose(fp);
    }

    return 0;
}