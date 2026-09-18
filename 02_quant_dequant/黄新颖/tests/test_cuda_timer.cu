#include <cuda_runtime_api.h>

#include <cmath>
#include <exception>
#include <iostream>
#include <string_view>

#include "common/cuda_stream.cuh"
#include "common/cuda_timer.cuh"

namespace {

/**
 * @brief 判断当前进程是否能创建 CUDA event 并在 default stream 中记录它们。
 *
 * @return 存在可用 CUDA device 时返回 true；无 device/driver 时打印跳过信息并
 * 返回 false；其他 runtime 错误同样返回 false，但调用方会按错误处理。
 */
[[nodiscard]] bool has_cuda_device() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && device_count == 0)) {
        std::cout << "CUDA runtime 不可用，跳过 CUDA Event 计时器测试。\n";
        return false;
    }
    if (status != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount 失败：" << cudaGetErrorString(status) << '\n';
        return false;
    }

    return true;
}

/**
 * @brief 验证独占 CudaStream 与其上 CudaEventTimer 的状态机和空区间计时。
 *
 * @return 非法调用被拒绝、正常 start/stop/elapsed 可重复执行时返回 true。
 */
[[nodiscard]] bool test_cuda_event_timer() {
    using quant_dequant::common::CudaStream;
    using quant_dequant::common::CudaEventTimer;
    using quant_dequant::common::CudaTimerError;

    try {
        // pipeline 使用的也是这对对象：stream 先拥有非阻塞 stream，再把借出的
        // handle 交给 event。此处验证二者的生命周期和顺序可以安全配合。
        CudaStream stream{};
        if (stream.get() == nullptr) {
            return false;
        }
        CudaEventTimer timer{stream.get()};
        try {
            static_cast<void>(timer.elapsedMilliseconds());
            return false;
        } catch (const CudaTimerError&) {
            // 未 stop 时读取 elapsed 是明确的状态机错误。
        }

        timer.start();
        try {
            timer.start();
            return false;
        } catch (const CudaTimerError&) {
            // running 状态不允许嵌套 start。
        }
        timer.stop();
        const float first_elapsed_ms = timer.elapsedMilliseconds();
        if (!std::isfinite(first_elapsed_ms) || first_elapsed_ms < 0.0F) {
            return false;
        }

        // 同一对象完成一个区间后可以安全复用；这正是 app 先量化、再反量化时
        // 希望复用 event 对而不重复分配的行为。
        timer.start();
        timer.stop();
        const float second_elapsed_ms = timer.elapsedMilliseconds();
        stream.synchronize();
        return std::isfinite(second_elapsed_ms) && second_elapsed_ms >= 0.0F;
    } catch (const std::exception& error) {
        std::cerr << "CUDA Event 计时器测试意外失败：" << error.what() << '\n';
        return false;
    }
}

}  // namespace

/**
 * @brief 运行 CUDA Event 计时器的 device 测试。
 *
 * @return 无 CUDA device/driver 时跳过并返回 0；有 device 时状态机和 event 计时
 * 正确返回 0；失败返回 1。
 */
int run_cuda_timer_tests() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && device_count == 0)) {
        std::cout << "CUDA runtime 不可用，跳过 CUDA Event 计时器测试。\n";
        return 0;
    }
    if (status != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount 失败：" << cudaGetErrorString(status) << '\n';
        return 1;
    }

    if (!has_cuda_device() || !test_cuda_event_timer()) {
        return 1;
    }

    return 0;
}
