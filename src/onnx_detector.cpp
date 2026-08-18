#include "detector.hpp"

#include <stdexcept>
#include <algorithm>

namespace yolo {

OnnxDetector::OnnxDetector()
    : env_(ORT_LOGGING_LEVEL_WARNING, "yolo_inference")
    , allocator_()
    , model_name_(kModelName)
    , loaded_(false)
    , input_tensor_size_(0)
    , output_tensor_size_(0) {
}

OnnxDetector::~OnnxDetector() = default;

bool OnnxDetector::loadModel(const std::string& model_path) {
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_BASIC);

        session_ = Ort::Session(env_, model_path.c_str(), session_options);
        loaded_ = true;

        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_input_nodes = session_.GetInputCount();
        if (num_input_nodes != 1) {
            throw std::runtime_error(
                "Expected 1 input node, got " + std::to_string(num_input_nodes));
        }

        {
            auto name_span = session_.GetInputNameAllocated(0, allocator);
            input_node_name_ = std::string(name_span.get());
            auto type_info = session_.GetInputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shape_ = tensor_info.GetShape();
            input_tensor_size_ = tensor_info.GetElementCount();
        }

        size_t num_output_nodes = session_.GetOutputCount();
        if (num_output_nodes != 1) {
            throw std::runtime_error(
                "Expected 1 output node, got " + std::to_string(num_output_nodes));
        }

        {
            auto name_span = session_.GetOutputNameAllocated(0, allocator);
            output_node_name_ = std::string(name_span.get());
            auto type_info = session_.GetOutputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_shape_ = tensor_info.GetShape();
            output_tensor_size_ = tensor_info.GetElementCount();
        }

        if (input_shape_.size() != 4 || input_shape_[0] != 1 ||
            input_shape_[1] != 3 || input_shape_[2] != 640 || input_shape_[3] != 640) {
            throw std::runtime_error(
                "Expected input shape [1, 3, 640, 640], got [" +
                std::to_string(input_shape_[0]) + ", " +
                std::to_string(input_shape_[1]) + ", " +
                std::to_string(input_shape_[2]) + ", " +
                std::to_string(input_shape_[3]) + "]");
        }

        if (output_shape_.size() != 3 || output_shape_[0] != 1 ||
            output_shape_[1] != 84 || output_shape_[2] != 8400) {
            throw std::runtime_error(
                "Expected output shape [1, 84, 8400], got [" +
                std::to_string(output_shape_[0]) + ", " +
                std::to_string(output_shape_[1]) + ", " +
                std::to_string(output_shape_[2]) + "]");
        }

    } catch (const std::exception& e) {
        loaded_ = false;
        model_name_ = "";
        throw std::runtime_error(std::string("Failed to load model: ") + e.what());
    }

    return true;
}

bool OnnxDetector::isLoaded() const {
    return loaded_;
}

std::vector<float> OnnxDetector::infer(const std::vector<float>& input_tensor) {
    if (!loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    if (input_tensor.size() != input_tensor_size_) {
        throw std::invalid_argument(
            "Input tensor size mismatch: " + std::to_string(input_tensor.size()) +
            ", expected " + std::to_string(input_tensor_size_));
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtDeviceAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor_value = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input_tensor.data()),
        input_tensor.size(),
        input_shape_.data(),
        input_shape_.size());

    const char* input_names[] = {input_node_name_.c_str()};
    const char* output_names[] = {output_node_name_.c_str()};

    std::vector<Ort::Value> output_tensors = session_.Run(
        Ort::RunOptions{},
        input_names,
        &input_tensor_value,
        1,
        output_names,
        1);

    if (output_tensors.empty()) {
        throw std::runtime_error("No output from model");
    }

    float* float_data = output_tensors[0].GetTensorMutableData<float>();
    std::vector<float> output(float_data, float_data + output_tensor_size_);

    return output;
}

} // namespace yolo
