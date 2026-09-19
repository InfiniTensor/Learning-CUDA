#pragma once

#include <acl/acl.h>
#include <acl/acl_rt.h>

namespace low_precision::ascend {

aclError initialize();
aclrtStream stream();
const char* error_string(aclError status);

}  // namespace low_precision::ascend
