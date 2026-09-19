#include "low_precision/ascend_runtime.hpp"

#include <mutex>

namespace low_precision::ascend {
namespace {

std::once_flag initialize_once;
aclError initialize_status = ACL_SUCCESS;
aclrtStream default_stream = nullptr;

void initialize_runtime() {
    initialize_status = aclInit(nullptr);
    if (initialize_status != ACL_SUCCESS) {
        return;
    }
    initialize_status = aclrtSetDevice(0);
    if (initialize_status != ACL_SUCCESS) {
        return;
    }
    initialize_status = aclrtCreateStream(&default_stream);
}

}  // namespace

aclError initialize() {
    std::call_once(initialize_once, initialize_runtime);
    return initialize_status;
}

aclrtStream stream() {
    return default_stream;
}

const char* error_string(aclError status) {
    if (status == ACL_SUCCESS) {
        return "success";
    }
    const char* recent_error = aclGetRecentErrMsg();
    return recent_error == nullptr ? "Ascend runtime error" : recent_error;
}

}  // namespace low_precision::ascend
