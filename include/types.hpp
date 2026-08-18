#pragma once

#include <string>
#include <vector>

namespace yolo {

struct BBox {
    float x;
    float y;
    float width;
    float height;
};

struct Detection {
    int class_id;
    std::string class_name;
    float confidence;
    BBox bbox;
};

struct ImageResult {
    std::string filename;
    std::string status;
    float inference_ms;
    std::vector<Detection> detections;
    std::string error;
};

struct BatchInferenceResult {
    std::string model_name;
    float confidence_threshold;
    float iou_threshold;
    std::vector<ImageResult> results;
};

struct PreprocessResult {
    std::vector<float> tensor;
    int original_width;
    int original_height;
    float scale;
    float pad_w;
    float pad_h;
};

} // namespace yolo
