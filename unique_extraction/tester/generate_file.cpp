#include <iostream>
#include <random>
#include <cstdint>


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

    FILE* fp;

    fp = std::fopen("./tester/input.in", "w");

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;

    for (uint64_t i = 0; i < data_size; i ++)
        fprintf(fp, "%llu ", dis(gen) % (1ULL << 30));

    fclose(fp);

    return 0;
}