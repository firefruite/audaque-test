#pragma once

#include "types.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

namespace yolo {

class OnnxDetector {
public:
    OnnxDetector();
    ~OnnxDetector();

    bool loadModel(const std::string& model_path);
    bool isLoaded() const;

    std::vector<float> infer(const std::vector<float>& input_tensor);

    const std::string& getModelName() const { return model_name_; }
    const std::vector<int64_t>& getInputShape() const { return input_shape_; }
    const std::vector<int64_t>& getOutputShape() const { return output_shape_; }

    static constexpr const char* kModelName = "yolo11n";

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string model_name_;
    bool loaded_;

    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    size_t input_tensor_size_;
    size_t output_tensor_size_;

    std::string input_node_name_;
    std::string output_node_name_;
};

} // namespace yolo
