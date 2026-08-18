# Finds a prebuilt ONNX Runtime distribution supplied by ONNXRUNTIME_ROOT.
# Expected layout: include/onnxruntime_cxx_api.h and lib/libonnxruntime.dylib.

if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()

find_path(OnnxRuntime_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS "${ONNXRUNTIME_ROOT}"
  PATH_SUFFIXES include include/onnxruntime/core/session)

find_library(OnnxRuntime_LIBRARY
  NAMES onnxruntime
  HINTS "${ONNXRUNTIME_ROOT}"
  PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OnnxRuntime
  REQUIRED_VARS OnnxRuntime_INCLUDE_DIR OnnxRuntime_LIBRARY)

if(OnnxRuntime_FOUND AND NOT TARGET OnnxRuntime::OnnxRuntime)
  add_library(OnnxRuntime::OnnxRuntime SHARED IMPORTED)
  set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
    IMPORTED_LOCATION "${OnnxRuntime_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${OnnxRuntime_INCLUDE_DIR}")
endif()
