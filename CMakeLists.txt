cmake_minimum_required(VERSION 3.20)

project(audaque_yolo11n LANGUAGES CXX)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT ONNXRUNTIME_ROOT)
  if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
  else()
    message(FATAL_ERROR
      "ONNXRUNTIME_ROOT is not set. "
      "Install ONNX Runtime (see scripts/setup-onnxruntime-macos-arm64.sh) "
      "and set the ONNXRUNTIME_ROOT environment variable.")
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(OnnxRuntime REQUIRED)

find_package(OpenCV REQUIRED)

find_package(nlohmann_json 3.11 REQUIRED)

set(CONDA_PREFIX "$ENV{CONDA_PREFIX}")
if(CONDA_PREFIX)
  list(APPEND CMAKE_PREFIX_PATH "${CONDA_PREFIX}")
endif()

add_library(yolo_core STATIC
  src/image_preprocessor.cpp
  src/postprocessor.cpp
  src/onnx_detector.cpp
  src/api_validation.cpp
  src/http_server.cpp
)
target_include_directories(yolo_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CONDA_PREFIX}/include
)
target_link_libraries(yolo_core PUBLIC
  OnnxRuntime::OnnxRuntime
  ${OpenCV_LIBS}
  nlohmann_json::nlohmann_json
)

add_executable(yolo_server src/main.cpp)
target_link_libraries(yolo_server PRIVATE yolo_core)

include(CTest)
if(BUILD_TESTING)
  enable_testing()

  add_executable(test_preprocessor tests/test_preprocessor.cpp)
  target_link_libraries(test_preprocessor PRIVATE yolo_core)
  add_test(NAME test_preprocessor COMMAND test_preprocessor)

  add_executable(test_postprocessor tests/test_postprocessor.cpp)
  target_link_libraries(test_postprocessor PRIVATE yolo_core)
  add_test(NAME test_postprocessor COMMAND test_postprocessor)

  add_executable(test_api_validation tests/test_api_validation.cpp)
  target_link_libraries(test_api_validation PRIVATE yolo_core)
  add_test(NAME test_api_validation COMMAND test_api_validation)

  add_executable(onnxruntime_smoke tests/onnxruntime_smoke.cpp)
  target_link_libraries(onnxruntime_smoke PRIVATE yolo_core)
  add_test(NAME onnxruntime_smoke COMMAND onnxruntime_smoke)
endif()

message(STATUS "ONNX Runtime root: ${ONNXRUNTIME_ROOT}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
