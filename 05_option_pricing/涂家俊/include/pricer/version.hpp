#pragma once

// 暴露命令行显示用的项目版本字符串。

namespace pricer {

// 返回稳定的版本文本，不分配内存也不会抛出异常。
const char *version() noexcept;

} // namespace pricer
