#pragma once

#include "types.hpp"
#include <vector>
#include <string>

namespace yolo {

class Postprocessor {
public:
    static constexpr int kNumClasses = 80;
    static constexpr int kOutputChannels = 84;

    Postprocessor() = default;

    std::vector<Detection> postprocess(
        const std::vector<float>& output,
        const PreprocessResult& pp_result,
        float confidence_threshold,
        float iou_threshold,
        const std::vector<std::string>& class_names
    );

    static float calculateIoU(const BBox& a, const BBox& b);

    static std::vector<int> nms(std::vector<Detection>& detections, float iou_threshold);
};

} // namespace yolo
