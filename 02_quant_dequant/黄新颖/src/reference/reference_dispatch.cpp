#include "quant_dequant/quantize.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <utility>

#include "reference_detail.hpp"

namespace quant_dequant {
namespace {

/**
 * @brief 验证 CPU reference 量化入口使用的 host 张量。
 *
 * @param input 待量化的 host 张量。
 * @throws ReferenceError shape、输入 dtype 或 FP32 values 长度不合法时抛出。
 */
void validate_reference_quantization_input(const HostTensor& input) {
    const auto element_count = input.desc.elementCount();
    if (!input.desc.isValid() || !is_supported_input_dtype(input.desc.dtype) ||
        !element_count.has_value() ||
        *element_count > std::numeric_limits<std::size_t>::max() ||
        input.values.size() != static_cast<std::size_t>(*element_count)) {
        throw ReferenceError{
            "CPU reference 量化要求合法的 FP16/FP32 输入描述和等长的 FP32 values。"};
    }
}

}  // namespace

ReferenceError::ReferenceError(std::string detail)
    : std::runtime_error(std::move(detail)) {}

QuantizedTensor quantize_reference(const HostTensor& input,
                                   const QuantizationConfig& config) {
    validate_reference_quantization_input(input);
    if (!config.isValid()) {
        throw ReferenceError{"CPU reference 量化配置不合法。"};
    }

    switch (config.format) {
        case QuantFormat::kMxfp8:
            return reference_detail::quantize_mxfp8_reference(input, config);

        case QuantFormat::kNvfp4:
            return reference_detail::quantize_nvfp4_reference(input, config);

        case QuantFormat::kUnknown:
            break;
    }

    throw ReferenceError{"CPU reference 量化格式不合法。"};
}

HostTensor dequantize_reference(const QuantizedTensor& input,
                                const DequantizationConfig& config) {
    if (!input.isConsistent()) {
        throw ReferenceError{"CPU reference 反量化要求自洽的 QuantizedTensor。"};
    }

    if (!config.isValid()) {
        throw ReferenceError{"CPU reference 反量化配置不合法。"};
    }

    switch (input.desc.format) {
        case QuantFormat::kMxfp8:
            return reference_detail::dequantize_mxfp8_reference(input, config);

        case QuantFormat::kNvfp4:
            return reference_detail::dequantize_nvfp4_reference(input, config);

        case QuantFormat::kUnknown:
            break;
    }

    throw ReferenceError{"CPU reference 反量化格式不合法。"};
}

}  // namespace quant_dequant
