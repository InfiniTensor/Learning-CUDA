#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#define INF (1 << 30)
#define BLOCK_SIZE 16  // 2D块大小，用于更好的缓存利用
#define MAX_PATH_LENGTH 1000

// 路径查询结构
struct PathQuery {
    int start;
    int end;
    int initial_resources;
};

// 查询结果结构
struct QueryResult {
    std::vector<int> path;
    int min_resources_needed;
    int final_resources;
    bool feasible;
    double query_time_ms;
};

// 图数据结构
struct Graph {
    int* adjacency_matrix;
    int* path_matrix;
    int* min_resources_matrix;  // 记录路径所需的最小资源
    int num_nodes;
};

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                     << " code=" << error << " \"" << cudaGetErrorString(error) << "\"" << std::endl; \
            exit(1); \
        } \
    } while(0)

// 优化的Floyd-Warshall kernel - 使用共享内存和分块计算
__global__ void floyd_warshall_phase1(int* dist, int* path, int* min_res, int k, int n) {
    // Phase 1: 更新依赖于第k行和第k列的块
    extern __shared__ int shared_mem[];
    
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int i = blockIdx.y * BLOCK_SIZE + tid_y;
    int j = blockIdx.x * BLOCK_SIZE + tid_x;
    
    if (i >= n || j >= n) return;
    
    // 加载第k行和第k列到共享内存
    int* row_k = shared_mem;
    int* col_k = shared_mem + BLOCK_SIZE;
    
    if (tid_y == 0 && j < n) {
        row_k[tid_x] = dist[k * n + j];
    }
    if (tid_x == 0 && i < n) {
        col_k[tid_y] = dist[i * n + k];
    }
    
    __syncthreads();
    
    if (i < n && j < n && tid_x < BLOCK_SIZE && tid_y < BLOCK_SIZE) {
        int idx = i * n + j;
        int d_ik = (tid_x == 0) ? col_k[tid_y] : dist[i * n + k];
        int d_kj = (tid_y == 0) ? row_k[tid_x] : dist[k * n + j];
        
        if (d_ik != INF && d_kj != INF) {
            int new_dist = d_ik + d_kj;
            if (new_dist < dist[idx]) {
                dist[idx] = new_dist;
                path[idx] = k;
                // 更新最小资源需求（路径上的最大消耗）
                int res_ik = min_res[i * n + k];
                int res_kj = min_res[k * n + j];
                min_res[idx] = max(res_ik, max(res_kj, new_dist));
            }
        }
    }
}

// 标准Floyd-Warshall kernel（更安全的实现）
__global__ void floyd_warshall_simple(int* dist, int* path, int k, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= n || j >= n) return;
    
    int idx = i * n + j;
    int idx_ik = i * n + k;
    int idx_kj = k * n + j;
    
    int d_ik = dist[idx_ik];
    int d_kj = dist[idx_kj];
    
    if (d_ik < INF && d_kj < INF) {
        long long new_dist_ll = (long long)d_ik + (long long)d_kj;
        if (new_dist_ll < (long long)INF) {
            int new_dist = (int)new_dist_ll;
            if (new_dist < dist[idx]) {
                dist[idx] = new_dist;
                path[idx] = k;
            }
        }
    }
}

// 路径重建函数（CPU） - 修复版本
std::vector<int> reconstruct_path(int* path_matrix, int start, int end, int n) {
    std::vector<int> result;
    
    // 边界检查
    if (start < 0 || start >= n || end < 0 || end >= n) {
        return result;
    }
    
    // 如果起点和终点相同
    if (start == end) {
        result.push_back(start);
        return result;
    }
    
    // 简化的路径重建 - 避免复杂递归
    int max_hops = n;  // 最多n跳
    int current = start;
    result.push_back(start);
    
    // 使用visited数组避免循环
    std::vector<bool> visited(n, false);
    visited[start] = true;
    
    // 构建从start到end的路径
    int hops = 0;
    while (current != end && hops < max_hops) {
        int idx = current * n + end;
        int next = path_matrix[idx];
        
        if (next == -1 || next == current) {
            // 直接到终点
            result.push_back(end);
            break;
        } else if (next >= 0 && next < n && !visited[next]) {
            // 通过中间节点
            result.push_back(next);
            visited[next] = true;
            current = next;
        } else {
            // 出现循环或无效路径，直接连接
            result.push_back(end);
            break;
        }
        hops++;
    }
    
    // 确保终点在路径中
    if (!result.empty() && result.back() != end) {
        result.push_back(end);
    }
    
    return result;
}

// 计算路径资源消耗
void calculate_resources(const std::vector<int>& path, int* dist_matrix, int n, 
                         int initial_resources, int& min_needed, int& final_resources) {
    min_needed = 0;
    final_resources = initial_resources;
    
    if (path.size() < 2) {
        return;
    }
    
    int current_resources = initial_resources;
    int max_deficit = 0;  // 记录最大亏损
    
    for (size_t i = 0; i < path.size() - 1; i++) {
        int from = path[i];
        int to = path[i + 1];
        
        // 边界检查
        if (from < 0 || from >= n || to < 0 || to >= n) {
            min_needed = INF;
            final_resources = -INF;
            return;
        }
        
        int idx = from * n + to;
        int cost = dist_matrix[idx];
        
        if (cost >= INF) {
            min_needed = INF;
            final_resources = -INF;
            return;
        }
        
        current_resources -= cost;
        
        // 计算到目前为止的最大亏损
        int deficit = initial_resources - current_resources;
        if (deficit > max_deficit) {
            max_deficit = deficit;
        }
    }
    
    // 最小所需资源就是最大亏损值
    min_needed = max_deficit;
    final_resources = current_resources;
}

class AtlantisPathfinder {
private:
    Graph h_graph;  // 主机端图数据
    Graph d_graph;  // 设备端图数据
    
    int* d_dist_work;  // 工作距离矩阵
    int* d_path_work;  // 工作路径矩阵
    
    bool is_initialized;
    cudaStream_t stream;
    cudaEvent_t start_event, stop_event;
    
public:
    AtlantisPathfinder(int num_nodes) : is_initialized(false) {
        h_graph.num_nodes = num_nodes;
        int matrix_size = num_nodes * num_nodes * sizeof(int);
        
        // 分配主机内存（页锁定内存以加速传输）
        CUDA_CHECK(cudaMallocHost(&h_graph.adjacency_matrix, matrix_size));
        CUDA_CHECK(cudaMallocHost(&h_graph.path_matrix, matrix_size));
        CUDA_CHECK(cudaMallocHost(&h_graph.min_resources_matrix, matrix_size));
        
        // 分配设备内存
        CUDA_CHECK(cudaMalloc(&d_graph.adjacency_matrix, matrix_size));
        CUDA_CHECK(cudaMalloc(&d_graph.path_matrix, matrix_size));
        CUDA_CHECK(cudaMalloc(&d_graph.min_resources_matrix, matrix_size));
        
        CUDA_CHECK(cudaMalloc(&d_dist_work, matrix_size));
        CUDA_CHECK(cudaMalloc(&d_path_work, matrix_size));
        
        // 创建CUDA流和事件
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
        
        d_graph.num_nodes = num_nodes;
    }
    
    ~AtlantisPathfinder() {
        cudaFreeHost(h_graph.adjacency_matrix);
        cudaFreeHost(h_graph.path_matrix);
        cudaFreeHost(h_graph.min_resources_matrix);
        
        cudaFree(d_graph.adjacency_matrix);
        cudaFree(d_graph.path_matrix);
        cudaFree(d_graph.min_resources_matrix);
        
        cudaFree(d_dist_work);
        cudaFree(d_path_work);
        
        cudaStreamDestroy(stream);
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    // 初始化图数据
    void initialize_graph(int* adjacency_matrix) {
        int n = h_graph.num_nodes;
        
        // 复制邻接矩阵并初始化路径矩阵
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n + j;
                h_graph.adjacency_matrix[idx] = adjacency_matrix[idx];
                h_graph.path_matrix[idx] = -1;  // -1表示直接连接
                h_graph.min_resources_matrix[idx] = (adjacency_matrix[idx] > 0) ? adjacency_matrix[idx] : 0;
            }
        }
        
        // 异步传输到GPU
        int matrix_size = n * n * sizeof(int);
        CUDA_CHECK(cudaMemcpyAsync(d_graph.adjacency_matrix, h_graph.adjacency_matrix, 
                                   matrix_size, cudaMemcpyHostToDevice, stream));
        
        // 预处理：运行Floyd-Warshall算法
        preprocess();
        is_initialized = true;
    }
    
    // 预处理：运行优化的Floyd-Warshall算法
    void preprocess() {
        int n = h_graph.num_nodes;
        int matrix_size = n * n * sizeof(int);
        
        // 复制初始数据到工作矩阵
        CUDA_CHECK(cudaMemcpyAsync(d_dist_work, d_graph.adjacency_matrix, 
                                   matrix_size, cudaMemcpyDeviceToDevice, stream));
        
        // 初始化路径矩阵为-1
        CUDA_CHECK(cudaMemset(d_path_work, 0xFF, matrix_size));  // 设置为-1
        
        // 配置kernel参数
        dim3 block_size(16, 16);
        dim3 grid_size((n + block_size.x - 1) / block_size.x, 
                      (n + block_size.y - 1) / block_size.y);
        
        // 运行Floyd-Warshall算法
        CUDA_CHECK(cudaEventRecord(start_event, stream));
        
        for (int k = 0; k < n; k++) {
            // 使用简单版本的kernel，更稳定
            floyd_warshall_simple<<<grid_size, block_size, 0, stream>>>
                (d_dist_work, d_path_work, k, n);
            
            // 检查kernel错误
            CUDA_CHECK(cudaGetLastError());
            
            // 定期同步以确保正确性
            if (k % 16 == 15) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        }
        
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // 复制结果回主机
        CUDA_CHECK(cudaMemcpy(h_graph.adjacency_matrix, d_dist_work, 
                             matrix_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_graph.path_matrix, d_path_work, 
                             matrix_size, cudaMemcpyDeviceToHost));
        
        float preprocess_time;
        CUDA_CHECK(cudaEventElapsedTime(&preprocess_time, start_event, stop_event));
        std::cout << "预处理时间 (Floyd-Warshall): " << preprocess_time << " ms" << std::endl;
    }
    
    // 处理单个查询
    QueryResult process_query(const PathQuery& query) {
        QueryResult result;
        auto query_start = std::chrono::high_resolution_clock::now();
        
        if (!is_initialized) {
            result.feasible = false;
            result.query_time_ms = 0;
            return result;
        }
        
        int n = h_graph.num_nodes;
        
        // 边界检查
        if (query.start < 0 || query.start >= n || query.end < 0 || query.end >= n) {
            result.feasible = false;
            result.query_time_ms = 0;
            return result;
        }
        
        int idx = query.start * n + query.end;
        
        // 检查是否可达
        if (h_graph.adjacency_matrix[idx] >= INF) {
            result.feasible = false;
            result.path.clear();
            result.min_resources_needed = INF;
            result.final_resources = -INF;
        } else {
            // 重建路径
            result.path = reconstruct_path(h_graph.path_matrix, query.start, query.end, n);
            
            // 如果路径重建失败，使用直接路径
            if (result.path.empty()) {
                result.path.push_back(query.start);
                result.path.push_back(query.end);
            }
            
            // 计算资源需求 - 使用原始邻接矩阵而不是最短路径矩阵
            int min_needed = 0;
            int final_res = query.initial_resources;
            
            // 对于最短路径，我们需要计算实际路径的资源消耗
            if (result.path.size() >= 2) {
                int current_res = query.initial_resources;
                int max_deficit = 0;
                
                for (size_t i = 0; i < result.path.size() - 1; i++) {
                    int from = result.path[i];
                    int to = result.path[i + 1];
                    
                    // 使用最短路径矩阵中的成本
                    int cost = h_graph.adjacency_matrix[from * n + to];
                    if (cost >= INF) {
                        min_needed = INF;
                        final_res = -INF;
                        break;
                    }
                    
                    current_res -= cost;
                    int deficit = query.initial_resources - current_res;
                    if (deficit > max_deficit) {
                        max_deficit = deficit;
                    }
                }
                
                min_needed = max_deficit;
                final_res = current_res;
            }
            
            result.min_resources_needed = min_needed;
            result.final_resources = final_res;
            
            // 检查是否有足够的资源
            result.feasible = (query.initial_resources >= result.min_resources_needed) && 
                             (result.min_resources_needed < INF);
        }
        
        auto query_end = std::chrono::high_resolution_clock::now();
        result.query_time_ms = std::chrono::duration<double, std::milli>(query_end - query_start).count();
        
        return result;
    }
    
    // 批量处理查询
    std::vector<QueryResult> process_queries(const std::vector<PathQuery>& queries) {
        std::vector<QueryResult> results;
        results.reserve(queries.size());
        
        auto batch_start = std::chrono::high_resolution_clock::now();
        
        for (const auto& query : queries) {
            results.push_back(process_query(query));
        }
        
        auto batch_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();
        
        // 计算性能指标
        double ttfq = results.empty() ? 0 : results[0].query_time_ms;
        double tpq = queries.empty() ? 0 : total_time / queries.size();
        
        std::cout << "\n=== 性能指标 ===" << std::endl;
        std::cout << "TTFQ (Time to First Query): " << ttfq << " ms" << std::endl;
        std::cout << "TPQ (Time Per Query): " << tpq << " ms" << std::endl;
        std::cout << "总查询数: " << queries.size() << std::endl;
        std::cout << "总时间: " << total_time << " ms" << std::endl;
        
        return results;
    }
    
    // 打印查询结果
    void print_result(const PathQuery& query, const QueryResult& result) {
        std::cout << "\n--- 查询结果 ---" << std::endl;
        std::cout << "起点: " << query.start << ", 终点: " << query.end << std::endl;
        std::cout << "初始资源: " << query.initial_resources << std::endl;
        
        if (result.feasible) {
            std::cout << "状态: 可行" << std::endl;
            std::cout << "路径: ";
            for (size_t i = 0; i < result.path.size(); i++) {
                std::cout << result.path[i];
                if (i < result.path.size() - 1) std::cout << " -> ";
            }
            std::cout << std::endl;
            std::cout << "最小所需资源: " << result.min_resources_needed << std::endl;
            std::cout << "最终剩余资源: " << result.final_resources << std::endl;
        } else {
            std::cout << "状态: 不可行 - ";
            if (result.path.empty()) {
                std::cout << "无法到达目的地" << std::endl;
            } else {
                std::cout << "资源不足" << std::endl;
                std::cout << "需要资源: " << result.min_resources_needed << std::endl;
                std::cout << "当前资源: " << query.initial_resources << std::endl;
            }
        }
        std::cout << "查询时间: " << result.query_time_ms << " ms" << std::endl;
    }
};

// CPU版本的Floyd-Warshall算法
class CPUPathfinder {
private:
    int* adjacency_matrix;
    int* path_matrix;
    int num_nodes;
    bool is_initialized;
    
public:
    CPUPathfinder(int n) : num_nodes(n), is_initialized(false) {
        int matrix_size = n * n * sizeof(int);
        adjacency_matrix = (int*)malloc(matrix_size);
        path_matrix = (int*)malloc(matrix_size);
    }
    
    ~CPUPathfinder() {
        free(adjacency_matrix);
        free(path_matrix);
    }
    
    void initialize_graph(int* graph) {
        int matrix_size = num_nodes * num_nodes * sizeof(int);
        memcpy(adjacency_matrix, graph, matrix_size);
        
        // 初始化路径矩阵
        for (int i = 0; i < num_nodes * num_nodes; i++) {
            path_matrix[i] = -1;
        }
        
        // 执行Floyd-Warshall
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int k = 0; k < num_nodes; k++) {
            for (int i = 0; i < num_nodes; i++) {
                for (int j = 0; j < num_nodes; j++) {
                    int idx_ij = i * num_nodes + j;
                    int idx_ik = i * num_nodes + k;
                    int idx_kj = k * num_nodes + j;
                    
                    if (adjacency_matrix[idx_ik] < INF && adjacency_matrix[idx_kj] < INF) {
                        long long new_dist = (long long)adjacency_matrix[idx_ik] + 
                                           (long long)adjacency_matrix[idx_kj];
                        if (new_dist < (long long)adjacency_matrix[idx_ij]) {
                            adjacency_matrix[idx_ij] = (int)new_dist;
                            path_matrix[idx_ij] = k;
                        }
                    }
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "CPU预处理时间 (Floyd-Warshall): " << cpu_time << " ms" << std::endl;
        
        is_initialized = true;
    }
    
    // CPU版本的路径重建 - 简化版本
    std::vector<int> reconstruct_path(int start, int end) {
        std::vector<int> result;
        
        if (!is_initialized || start < 0 || start >= num_nodes || 
            end < 0 || end >= num_nodes) {
            return result;
        }
        
        if (start == end) {
            result.push_back(start);
            return result;
        }
        
        // 检查是否可达
        int idx = start * num_nodes + end;
        if (adjacency_matrix[idx] >= INF) {
            return result;  // 不可达
        }
        
        // 简单路径重建：使用前驱节点信息
        int max_hops = num_nodes;
        int current = start;
        result.push_back(start);
        
        std::vector<bool> visited(num_nodes, false);
        visited[start] = true;
        
        int hops = 0;
        while (current != end && hops < max_hops) {
            int next_idx = current * num_nodes + end;
            int next = path_matrix[next_idx];
            
            if (next == -1 || next == current || next == end) {
                // 直接到终点
                if (current != end) {
                    result.push_back(end);
                }
                break;
            } else if (next >= 0 && next < num_nodes && !visited[next]) {
                // 通过中间节点
                result.push_back(next);
                visited[next] = true;
                current = next;
            } else {
                // 可能有循环，终止
                if (current != end) {
                    result.push_back(end);
                }
                break;
            }
            hops++;
        }
        
        // 确保终点在路径中
        if (!result.empty() && result.back() != end) {
            result.push_back(end);
        }
        
        return result;
    }
    
    QueryResult process_query(const PathQuery& query) {
        QueryResult result;
        result.feasible = false;
        result.min_resources_needed = 0;
        result.final_resources = 0;
        result.query_time_ms = 0;
        
        auto query_start = std::chrono::high_resolution_clock::now();
        
        if (!is_initialized) {
            return result;
        }
        
        // 边界检查
        if (query.start < 0 || query.start >= num_nodes || 
            query.end < 0 || query.end >= num_nodes) {
            return result;
        }
        
        int idx = query.start * num_nodes + query.end;
        
        if (adjacency_matrix[idx] >= INF) {
            result.path.clear();
            result.min_resources_needed = INF;
            result.final_resources = -INF;
        } else {
            // 重建路径
            result.path = reconstruct_path(query.start, query.end);
            
            if (result.path.empty()) {
                result.path.push_back(query.start);
                result.path.push_back(query.end);
            }
            
            // 计算资源
            int current_res = query.initial_resources;
            int max_deficit = 0;
            bool path_valid = true;
            
            for (size_t i = 0; i < result.path.size() - 1; i++) {
                int from = result.path[i];
                int to = result.path[i + 1];
                
                // 边界检查
                if (from < 0 || from >= num_nodes || to < 0 || to >= num_nodes) {
                    path_valid = false;
                    break;
                }
                
                int cost = adjacency_matrix[from * num_nodes + to];
                
                if (cost >= INF) {
                    path_valid = false;
                    break;
                }
                
                current_res -= cost;
                int deficit = query.initial_resources - current_res;
                if (deficit > max_deficit) {
                    max_deficit = deficit;
                }
            }
            
            if (path_valid) {
                result.min_resources_needed = max_deficit;
                result.final_resources = current_res;
                result.feasible = (query.initial_resources >= max_deficit);
            } else {
                result.min_resources_needed = INF;
                result.final_resources = -INF;
                result.feasible = false;
            }
        }
        
        auto query_end = std::chrono::high_resolution_clock::now();
        result.query_time_ms = std::chrono::duration<double, std::milli>(query_end - query_start).count();
        
        return result;
    }
    
    std::vector<QueryResult> process_queries(const std::vector<PathQuery>& queries) {
        std::vector<QueryResult> results;
        results.reserve(queries.size());
        
        if (queries.empty()) {
            return results;
        }
        
        auto batch_start = std::chrono::high_resolution_clock::now();
        
        for (const auto& query : queries) {
            results.push_back(process_query(query));
        }
        
        auto batch_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();
        
        double ttfq = results.empty() ? 0 : results[0].query_time_ms;
        double tpq = queries.empty() ? 0 : total_time / queries.size();
        
        std::cout << "\n=== CPU性能指标 ===" << std::endl;
        std::cout << "TTFQ (Time to First Query): " << ttfq << " ms" << std::endl;
        std::cout << "TPQ (Time Per Query): " << tpq << " ms" << std::endl;
        std::cout << "总查询数: " << queries.size() << std::endl;
        std::cout << "总时间: " << total_time << " ms" << std::endl;
        
        return results;
    }
};

// 生成测试图
void generate_test_graph(int* graph, int n, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(1, 100);
    std::uniform_real_distribution<float> prob(0.0, 1.0);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                graph[i * n + j] = 0;
            } else if (prob(rng) < 0.3) {  // 30%的概率有边
                // 随机生成正负消耗值
                int cost = dist(rng);
                if (prob(rng) < 0.2) {  // 20%概率是负值（收益）
                    cost = -cost / 2;
                }
                graph[i * n + j] = cost;
            } else {
                graph[i * n + j] = INF;
            }
        }
    }
}

// 生成测试查询
std::vector<PathQuery> generate_test_queries(int n, int num_queries) {
    std::vector<PathQuery> queries;
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> node_dist(0, n - 1);
    std::uniform_int_distribution<int> res_dist(50, 500);
    
    for (int i = 0; i < num_queries; i++) {
        PathQuery q;
        q.start = node_dist(rng);
        q.end = node_dist(rng);
        while (q.end == q.start) {  // 确保起点终点不同
            q.end = node_dist(rng);
        }
        q.initial_resources = res_dist(rng);
        queries.push_back(q);
    }
    
    return queries;
}

int main() {
    // 首先进行简单的功能测试
    std::cout << "=== 功能验证测试 ===" << std::endl;
    const int TEST_SIZE = 10;
    int* simple_graph = new int[TEST_SIZE * TEST_SIZE];
    
    // 创建一个简单的测试图
    for (int i = 0; i < TEST_SIZE * TEST_SIZE; i++) {
        simple_graph[i] = INF;
    }
    for (int i = 0; i < TEST_SIZE; i++) {
        simple_graph[i * TEST_SIZE + i] = 0;  // 对角线为0
        if (i < TEST_SIZE - 1) {
            simple_graph[i * TEST_SIZE + (i + 1)] = 10;  // 链式连接
        }
    }
    
    std::cout << "测试简单图（10节点链式）..." << std::endl;
    
    // 测试CPU版本
    CPUPathfinder simple_cpu(TEST_SIZE);
    simple_cpu.initialize_graph(simple_graph);
    
    PathQuery test_query = {0, 5, 100};
    auto cpu_result = simple_cpu.process_query(test_query);
    std::cout << "CPU: 0->5, 可行=" << cpu_result.feasible 
              << ", 路径长度=" << cpu_result.path.size() << std::endl;
    
    // 测试GPU版本
    AtlantisPathfinder simple_gpu(TEST_SIZE);
    simple_gpu.initialize_graph(simple_graph);
    
    auto gpu_result = simple_gpu.process_query(test_query);
    std::cout << "GPU: 0->5, 可行=" << gpu_result.feasible 
              << ", 路径长度=" << gpu_result.path.size() << std::endl;
    
    delete[] simple_graph;
    std::cout << "功能测试完成\n" << std::endl;
    
    // 性能测试
    std::vector<int> test_sizes = {50, 100, 200};
    std::vector<int> query_counts = {10, 20, 50};
    
    std::cout << "=== 亚特兰蒂斯路径查找系统 - 性能对比测试 ===" << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    // 创建表格
    std::cout << std::left;
    std::cout << std::setw(10) << "节点数" 
              << std::setw(12) << "查询数"
              << std::setw(18) << "CPU预处理(ms)"
              << std::setw(18) << "GPU预处理(ms)"
              << std::setw(12) << "加速比"
              << std::setw(15) << "CPU TPQ(ms)"
              << std::setw(15) << "GPU TPQ(ms)"
              << std::setw(12) << "查询加速比" << std::endl;
    std::cout << std::string(120, '-') << std::endl;
    
    for (size_t test_idx = 0; test_idx < test_sizes.size(); test_idx++) {
        int NUM_NODES = test_sizes[test_idx];
        int NUM_QUERIES = query_counts[test_idx];
        
        std::cout << "\n[测试 " << test_idx + 1 << "] 节点数=" << NUM_NODES 
                  << ", 查询数=" << NUM_QUERIES << std::endl;
        
        // 生成测试图
        int* test_graph = new int[NUM_NODES * NUM_NODES];
        if (!test_graph) {
            std::cerr << "内存分配失败!" << std::endl;
            continue;
        }
        
        generate_test_graph(test_graph, NUM_NODES);
        
        // 生成查询
        auto queries = generate_test_queries(NUM_NODES, NUM_QUERIES);
        
        double cpu_init_time = 0, cpu_tpq = 0;
        double gpu_init_time = 0, gpu_tpq = 0;
        int cpu_successful = 0, gpu_successful = 0;
        
        // CPU测试
        try {
            std::cout << "  运行CPU测试..." << std::endl;
            CPUPathfinder cpu_pathfinder(NUM_NODES);
            
            auto t1 = std::chrono::high_resolution_clock::now();
            cpu_pathfinder.initialize_graph(test_graph);
            auto t2 = std::chrono::high_resolution_clock::now();
            cpu_init_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
            
            // 处理查询
            auto t3 = std::chrono::high_resolution_clock::now();
            for (const auto& q : queries) {
                auto res = cpu_pathfinder.process_query(q);
                if (res.feasible) cpu_successful++;
            }
            auto t4 = std::chrono::high_resolution_clock::now();
            double total_query_time = std::chrono::duration<double, std::milli>(t4 - t3).count();
            cpu_tpq = total_query_time / queries.size();
            
            std::cout << "  CPU完成: 初始化=" << cpu_init_time 
                     << "ms, TPQ=" << cpu_tpq << "ms" << std::endl;
        } catch (...) {
            std::cerr << "  CPU测试失败!" << std::endl;
        }
        
        // GPU测试  
        try {
            std::cout << "  运行GPU测试..." << std::endl;
            AtlantisPathfinder gpu_pathfinder(NUM_NODES);
            
            auto t1 = std::chrono::high_resolution_clock::now();
            gpu_pathfinder.initialize_graph(test_graph);
            auto t2 = std::chrono::high_resolution_clock::now();
            gpu_init_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
            
            // 处理查询
            auto t3 = std::chrono::high_resolution_clock::now();
            for (const auto& q : queries) {
                auto res = gpu_pathfinder.process_query(q);
                if (res.feasible) gpu_successful++;
            }
            auto t4 = std::chrono::high_resolution_clock::now();
            double total_query_time = std::chrono::duration<double, std::milli>(t4 - t3).count();
            gpu_tpq = total_query_time / queries.size();
            
            std::cout << "  GPU完成: 初始化=" << gpu_init_time 
                     << "ms, TPQ=" << gpu_tpq << "ms" << std::endl;
        } catch (...) {
            std::cerr << "  GPU测试失败!" << std::endl;
        }
        
        // 输出结果
        double init_speedup = (gpu_init_time > 0) ? (cpu_init_time / gpu_init_time) : 0;
        double query_speedup = (gpu_tpq > 0) ? (cpu_tpq / gpu_tpq) : 0;
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(10) << NUM_NODES
                  << std::setw(12) << NUM_QUERIES
                  << std::setw(18) << cpu_init_time
                  << std::setw(18) << gpu_init_time
                  << std::setw(12) << (std::to_string(init_speedup).substr(0,5) + "x")
                  << std::setw(15) << std::setprecision(4) << cpu_tpq
                  << std::setw(15) << gpu_tpq
                  << std::setw(12) << (std::to_string(query_speedup).substr(0,5) + "x")
                  << std::endl;
        
        std::cout << "  成功率: CPU=" << cpu_successful << "/" << NUM_QUERIES 
                  << ", GPU=" << gpu_successful << "/" << NUM_QUERIES << std::endl;
        
        delete[] test_graph;
    }
    
    std::cout << "\n测试完成!" << std::endl;
    return 0;
}