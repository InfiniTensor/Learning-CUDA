#include <omp.h>
#include <filesystem>
#include <regex>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <sys/mman.h>
#include <memory>
#include <random>
#include <iostream>

#define FILE_DATA_SIZE (1 << 24)
#define BLOCKSIZE 256

const uint64_t data_tile_size = 1 << 24;

/**
 * @author JBFYS-XD
 */

/**
 * @brief 对小于 tile 大小的 GPU 运算
 * 
 * @param d_data 待排数组
 * @param d_data_size 待排数组大小
 * @param k 步长， 以长度为 k 为单位进行排序
 * @param j 对长度 k 的每 j 个进行双调排序
 */
__global__ void bitonic_sort_tile_kernel(uint64_t* d_data, uint64_t d_data_size, uint64_t k, uint64_t j) {
    uint64_t x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x >= d_data_size || x & j) return;
    uint64_t y = j == (k >> 1) ? x + 2 * (j - (x % j)) - 1 : x ^ j;
    if (y >= d_data_size) return;
    uint64_t valx = d_data[x];
    uint64_t valy = d_data[y];
    if (valx > valy) {
        d_data[x] = valy;
        d_data[y] = valx;
    }
}

/**
 * @brief 进行步间分块为对单内核可完成（数据量可容）的数组，进行排序
 * 
 * @param data 分块的待排数组
 * @param d_data 加载到显存数组
 * @param d_data_size 待排数组大小
 */
void bitonic_sort_tile(uint64_t* data, uint64_t* d_data, uint64_t d_data_size) {
    
    cudaMemcpy(d_data, data, d_data_size * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int threadsPerBlock = BLOCKSIZE;
    int blocksPerGrid = (d_data_size + threadsPerBlock - 1) / threadsPerBlock;


    for (uint64_t k = 2; (k >> 1) < d_data_size; k <<= 1) {
        for (uint64_t j = (k >> 1); j > 0; j >>= 1) {
            bitonic_sort_tile_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_data_size, k, j);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(data, d_data, d_data_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
}


__global__ void bitonic_sort_merge_kernel(
    uint64_t* data1, uint64_t data1_size, 
    uint64_t* data2, uint64_t data2_size, uint64_t k, uint64_t j
) {
    uint64_t x = blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t y = j == (k >> 1) ? data_tile_size - x - 1 : x;
    if (x >= data_tile_size || x >= data1_size || y >= data2_size) return;

    uint64_t valx = data1[x];
    uint64_t valy = data2[y];
    if (valx > valy) {
        data1[x] = valy;
        data2[y] = valx;
    }
}

/**
 * @brief 对步长大于 data_TILE_SIZE 的进行步内分块
 * 
 * @param data 全特征数组
 * @param data_size data 元素数
 * @param d_data1 映射步内分块后的操作对的前一个数组
 * @param d_data2 映射步内分块后的操作对的后一个数组
 */
void bitonic_sort_merge(uint64_t* data, uint64_t data_size, uint64_t* d_data1, uint64_t* d_data2) {

    int threadsPerBlock = BLOCKSIZE;
    int blocksPerGrid = (data_tile_size + threadsPerBlock - 1) / threadsPerBlock;

    for (uint64_t k = 2 * data_tile_size; (k >> 1) < data_size; k <<= 1) {
        for (uint64_t j = (k >> 1); j >= data_tile_size; j >>= 1) {
            for (uint64_t idx = 0; idx < data_size; idx += data_tile_size) {
                if (idx & j) continue;
                uint64_t data1_l = idx;
                uint64_t data1_r = idx + data_tile_size - 1;
                uint64_t data2_l = j == (k >> 1) ? data1_r + 2 * (j - (data1_r % j)) - 1 : data1_l ^ j;
                uint64_t data2_r = j == (k >> 1) ? data1_l + 2 * (j - (data1_l % j)) - 1 : data1_r ^ j;
                if (data2_l >= data_size) continue;
                data2_r = std::min(data2_r, data_size - 1);
                uint64_t data1_len = data1_r - data1_l + 1;
                uint64_t data2_len = data2_r - data2_l + 1;
                cudaMemcpy(d_data1, data + data1_l, data1_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
                cudaMemcpy(d_data2, data + data2_l, data2_len * sizeof(uint64_t), cudaMemcpyHostToDevice);

                bitonic_sort_merge_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                    d_data1, data1_len,
                    d_data2, data2_len, k, j
                );
                cudaDeviceSynchronize();

                cudaMemcpy(data + data1_l, d_data1, data1_len * sizeof(uint64_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(data + data2_l, d_data2, data2_len * sizeof(uint64_t), cudaMemcpyDeviceToHost);

            }
        }
        for (uint64_t i = 0; i < data_size; i += data_tile_size) {
            uint64_t d_data1_size = std::min(data_tile_size, data_size - i);
            bitonic_sort_tile(data + i, d_data1, d_data1_size);
        }
    }
}

/**
 * @brief 使用双调排序实现全特征排序
 * 
 * @param data 待排序数组
 * @param data_size 待排序数组大小
 * 
 * @note 此处双调排序使用的是适用于任意长度的版本
 */

void bitonic_sort(uint64_t* data, uint64_t data_size, uint64_t* d_data1, uint64_t* d_data2) {

    // 先对每个 tile 块进行单独排序
    for (uint64_t i = 0; i < data_size; i += data_tile_size) {
        uint64_t d_data1_size = std::min(data_tile_size, data_size - i);
        bitonic_sort_tile(data + i, d_data1, d_data1_size);
    }

    // 对大于 data_tile_size 的进行步内分块排序
    bitonic_sort_merge(data, data_size, d_data1, d_data2);

}

__global__ void mark_tile_kernel(uint64_t* input, uint64_t data_size, uint64_t* output, uint64_t first_mark) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= data_size) return;
    if (idx == 0)
        output[idx] = first_mark;
    else
        output[idx] = input[idx] != input[idx - 1];
}

void mark_tile(uint64_t* data, uint64_t data_size, uint64_t* input, uint64_t* marks, uint64_t first_mark) {

    int threadsPerBlock = BLOCKSIZE;
    int blocksPerGrid = (data_size + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemcpy(input, data, data_size * sizeof(uint64_t), cudaMemcpyHostToDevice);

    mark_tile_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, data_size, marks, first_mark);
    cudaDeviceSynchronize();
}

__global__ void scan_tile_kernel(uint64_t* mark, uint64_t data_size, uint64_t i) {
    __shared__ uint64_t smem[BLOCKSIZE];
    
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    uint64_t now = (idx + 1) * i - 1;
    smem[tid] = now < data_size ? mark[now] : 0ULL;

    for (uint32_t i = 2; i <= BLOCKSIZE; i <<= 1) {
        __syncthreads();
        uint64_t pos = (tid + 1) * i - 1;
        if (pos < BLOCKSIZE)
            smem[pos] += smem[pos - (i >> 1)];
    }

    for (uint32_t i = BLOCKSIZE / 2; i > 1; i >>= 1) {
        __syncthreads();
        uint64_t pos = (tid + 1) * i - 1;
        if (pos + (i >> 1) < BLOCKSIZE)
            smem[pos + (i >> 1)] += smem[pos];
    }

    __syncthreads();
    if (now < data_size)
        mark[now] = smem[tid];
}

__global__ void scan_tile_sum_kernel(uint64_t* marks, uint64_t data_size, uint64_t* scan) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t val = marks[idx];
    for (uint64_t i = BLOCKSIZE; i < data_size; i *= BLOCKSIZE) {
        uint64_t now = (idx + 1) / i;
        if (now == 0 || ((idx + 1) % i) == 0) continue;
        val += marks[now * i - 1];
    }
    scan[idx] = val;
}

void scan_tile(uint64_t* marks, uint64_t data_size, uint64_t* scan) {

    int threadsPerBlock = BLOCKSIZE;

    for (uint64_t i = 1; i < data_size; i *= threadsPerBlock) {
        uint64_t oper_size = data_size / i;
        int blocksPerGrid = (oper_size + threadsPerBlock - 1) / threadsPerBlock;

        scan_tile_kernel<<<blocksPerGrid, threadsPerBlock>>>(marks, data_size, i);
        cudaDeviceSynchronize();

    }

    int blocksPerGrid = (data_size + threadsPerBlock - 1) / threadsPerBlock;
    scan_tile_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(marks, data_size, scan);
    cudaDeviceSynchronize();
}

__global__ void write_tile_kernel(uint64_t* data, uint64_t* scan, uint64_t data_size, uint64_t* write) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= data_size) return;
    if (scan[idx] >= 1)
        write[scan[idx] - 1] = data[idx];
}

void write_tile(
    uint64_t* data, uint64_t* scan, uint64_t data_size, 
    uint64_t* write, uint64_t* result, uint64_t& result_size
) {
    cudaMemcpy(&result_size, scan + data_size - 1, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    int threadsPerBlock = BLOCKSIZE;
    int blocksPerGrid = (data_size + threadsPerBlock - 1) / threadsPerBlock;

    write_tile_kernel<<<blocksPerGrid, threadsPerBlock>>>(data, scan, data_size, write);
    cudaDeviceSynchronize();

    cudaMemcpy(result, write, result_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
}

/**
 * @brief 用于对大数据特征的去重, 总调度函数
 * 
 * @param input_path 特征文件路径
 * @param output_path 去重后文件路径
 * 
 * @note 解决方案: 全特征排序-->分块标记独特特征-->分块scan前缀和标记
 *          -->分块以前缀和标记下标写回独特特征-->分块写入output_path
 */
void unique_extraction(std::string input_path, std::string output_path = "unique_feature.output") {
    

    uint64_t data_size = 0;
    size_t file_num = 0;

    std::string folderPath = input_path;
    std::regex filePattern(R"(uint64-(\d+)-(\d+)\.in)");

    std::vector<std::string> filename;
    std::vector<uint64_t> data_len;

    try {
        for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
            if (!entry.is_regular_file()) continue;
            
            filename.push_back(entry.path().filename().string());

            std::smatch matches;

            if (std::regex_match(filename[file_num], matches, filePattern)) {
                data_len.push_back(std::stoull(matches[1].str()));
                data_size += data_len[file_num];

            }
            file_num ++;
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << "\n";
    }

    std::cout << "number of data: " << data_size << "\n";
    std::cout << "number of files: " << file_num << "\n";

    std::cout << "Max threads: " << omp_get_max_threads() << "\n";

    std::vector<uint64_t> data(data_size);

    uint64_t file_cnt = 0;

    bool file_check = true;

    #pragma omp parallel for
    for (int i = 0; i < file_num; i++) {
        FILE *fp_input;
        fp_input = std::fopen((folderPath + filename[i]).c_str(), "r");

        if (!fp_input) {
            std::cout << "false to open file " << filename[i] << "\n";
            continue;
        }

        uint64_t it = 0;
        #pragma omp atomic capture
        {
            it = file_cnt;
            file_cnt += data_len[i];
        }
        
        uint64_t tmp, check = 0;
        while (std::fscanf(fp_input, "%llu", &tmp) != EOF) {
            data[it++] = tmp;
            check ++;
        }
        if (check != data_len[i]) {
            file_check = false;
        }
        
        std::fclose(fp_input);
        
    }
    
    if (!file_check || file_cnt != data_size) {
        std::cout << file_check << " " << file_cnt << "\n";
        std::cerr << "false to read file\n";
    } else {
        std::cout << "success to read file\n";
    }
    
    uint64_t *d_data1, *d_data2, *d_data3;
    cudaMalloc(&d_data1, data_tile_size * sizeof(uint64_t));
    cudaMalloc(&d_data2, data_tile_size * sizeof(uint64_t));
    cudaMalloc(&d_data3, data_tile_size * sizeof(uint64_t));
    
    
    printf("sort of all element start\n");
    // 全特征排序
    bitonic_sort(data.data(), data_size, d_data1, d_data2);
    printf("sort of all element end\n");
    
    // auto result = std::make_unique<uint64_t[]>(data_tile_size);
    
    
    uint64_t end_size = 0;
    
    for (uint64_t i = 0; i < data_size; i += data_tile_size) {
        uint64_t tile_size = std::min(data_tile_size, data_size - i);
        
        
        // 用于判断分块的第一个元素是否 unique
        uint64_t first_mark = i == 0 || data[i - 1] != data[i];
        // 标记 unique feature
        mark_tile(data.data() + i, tile_size, d_data1, d_data2, first_mark);
        
        // scan_tile
        scan_tile(d_data2, tile_size, d_data3);

        uint64_t result_size;
        write_tile(d_data1, d_data3, tile_size, d_data2, data.data() + end_size, result_size);

        end_size += result_size;
        
        printf("tile range %llu finished\n", i / data_tile_size);
    }


    int out_file_num = (end_size + FILE_DATA_SIZE - 1) / FILE_DATA_SIZE;
    
    std::cout << "number of out element: " << end_size << "\n";
    std::cout << "number of out file: " << out_file_num << "\n";

    #pragma omp parallel for 
    for (int i = 0; i < out_file_num; i++) {

        int tid = omp_get_thread_num();
        uint64_t out_start = i * FILE_DATA_SIZE;
        int out_len = std::min((uint64_t)FILE_DATA_SIZE, end_size - out_start);
        char path[100];
        snprintf(path, 100 * sizeof(char), "./tester/output/uint64-%llu-%llu.out", out_len, i);
        
        FILE* fp;

        fp = std::fopen(path, "w");
        
        for (uint64_t j = 0; j < out_len; j ++)
            fprintf(fp, "%llu ", data[out_start + j]);
    
        fclose(fp);
    }

    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_data3);
}

// test
int main() {
    unique_extraction("./tester/input/", "./tester/output/");
    return 0;
}

// 381344736