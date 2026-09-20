set(_HADAMARD_METAX_ROOTS "")

if(DEFINED ENV{MACA_PATH})
  list(APPEND _HADAMARD_METAX_ROOTS "$ENV{MACA_PATH}")
endif()
list(APPEND _HADAMARD_METAX_ROOTS /opt/maca)

find_path(MetaX_ROOT
  NAMES include/mcr/mc_runtime.h
  PATHS ${_HADAMARD_METAX_ROOTS}
  NO_DEFAULT_PATH
)

find_program(MetaX_MXCC_EXECUTABLE
  NAMES mxcc
  PATHS "${MetaX_ROOT}/mxgpu_llvm/bin" /opt/maca/mxgpu_llvm/bin
)

find_path(MetaX_INCLUDE_DIR
  NAMES mcr/mc_runtime.h
  PATHS "${MetaX_ROOT}/include" /opt/maca/include
)

find_library(MetaX_MCRUNTIME_LIBRARY
  NAMES mcruntime
  PATHS "${MetaX_ROOT}/lib" /opt/maca/lib
)

find_library(MetaX_MXCRT_LIBRARY
  NAMES mxc-runtime64
  PATHS "${MetaX_ROOT}/lib" /opt/maca/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MetaX
  REQUIRED_VARS
    MetaX_ROOT
    MetaX_MXCC_EXECUTABLE
    MetaX_INCLUDE_DIR
    MetaX_MCRUNTIME_LIBRARY
)

if(MetaX_FOUND)
  set(MetaX_CUBRIDGE_ROOT "${MetaX_ROOT}/tools/cu-bridge")
  set(MetaX_CMAKE_MODULE_DIR "${MetaX_CUBRIDGE_ROOT}/cmake_module/maca")
  set(MetaX_INCLUDE_DIRS "${MetaX_INCLUDE_DIR}")
  set(MetaX_LIBRARIES "${MetaX_MCRUNTIME_LIBRARY}")
  if(MetaX_MXCRT_LIBRARY)
    list(APPEND MetaX_LIBRARIES "${MetaX_MXCRT_LIBRARY}")
  endif()
endif()

mark_as_advanced(
  MetaX_ROOT
  MetaX_MXCC_EXECUTABLE
  MetaX_INCLUDE_DIR
  MetaX_MCRUNTIME_LIBRARY
  MetaX_MXCRT_LIBRARY
)
