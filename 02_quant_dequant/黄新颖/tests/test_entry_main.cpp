/**
 * @brief 以编译期指定的单个测试入口构造独立测试可执行文件。
 *
 * 每个测试源文件继续只定义一个 `run_*_tests()` 函数；CMake 为它单独构建一个
 * 可执行文件，并通过 `QUANT_DEQUANT_TEST_ENTRY` 选择此处调用的函数。这样 CTest
 * 可以独立报告和筛选每个模块，同时避免为每个测试手写重复的 `main()`。
 */

#ifndef QUANT_DEQUANT_TEST_ENTRY
#error "每个测试可执行文件必须通过 CMake 定义 QUANT_DEQUANT_TEST_ENTRY。"
#endif

/** @brief 当前 CTest 可执行文件对应的唯一测试函数。 */
int QUANT_DEQUANT_TEST_ENTRY();

/**
 * @brief 执行由 CMake 注入的单个模块测试入口。
 *
 * @return 测试函数的进程退出码；0 表示通过，非 0 表示失败。
 */
int main() {
    return QUANT_DEQUANT_TEST_ENTRY();
}
