#pragma once

#include "vsearch/common.hpp"

namespace vs {

// Parse a retrieval-parameter file (text, key = value, '#' comments).
// Unknown keys are preserved in `extra` so driver tools can log them.
SearchConfig parseParamFile(const std::string& path);
SearchConfig parseParamString(const std::string& text);

}  // namespace vs
