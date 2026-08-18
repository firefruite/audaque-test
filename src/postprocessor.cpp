#include "postprocessor.hpp"
#include "image_preprocessor.hpp"

#include <algorithm>
#include <stdexcept>

namespace yolo {

float Postprocessor::calculateIoU(const BBox& a, const BBox& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);

    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;

    float area_a = a.width * a.height;
    float area_b = b.width * b.height;
    float union_area = area_a + area_b - inter;

    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return inter / union_area;
}

std::vector<int> Postprocessor::nms(std::vector<Detection>& detections, float iou_threshold) {
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<int> keep;

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        keep.push_back(static_cast<int>(i));

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            if (detections[i].class_id != detections[j].class_id) continue;

            float iou = calculateIoU(detections[i].bbox, detections[j].bbox);
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return keep;
}

std::vector<Detection> Postprocessor::postprocess(
    const std::vector<float>& output,
    const PreprocessResult& pp_result,
    float confidence_threshold,
    float iou_threshold,
    const std::vector<std::string>& class_names
) {
    if (output.size() != static_cast<size_t>(kOutputChannels * 8400)) {
        throw std::invalid_argument(
            "Unexpected output tensor size: " + std::to_string(output.size()) +
            ", expected " + std::to_string(kOutputChannels * 8400)
        );
    }

    std::vector<Detection> detections;

    const int num_anchors = 8400;

    for (int i = 0; i < num_anchors; ++i) {
        float cx = output[i];
        float cy = output[num_anchors + i];
        float w = output[2 * num_anchors + i];
        float h = output[3 * num_anchors + i];

        float best_class_conf = 0.0f;
        int best_class = -1;
        for (int c = 0; c < kNumClasses; ++c) {
            float class_conf = output[(4 + c) * num_anchors + i];
            if (class_conf > best_class_conf) {
                best_class_conf = class_conf;
                best_class = c;
            }
        }

        if (best_class < 0) continue;

        float confidence = best_class_conf;
        if (confidence < confidence_threshold) continue;

        BBox bbox_640;
        bbox_640.x = cx - w / 2.0f;
        bbox_640.y = cy - h / 2.0f;
        bbox_640.width = w;
        bbox_640.height = h;

        BBox bbox_orig = ImagePreprocessor::scaleBboxToOriginal(bbox_640, pp_result);

        Detection det;
        det.class_id = best_class;
        det.confidence = confidence;
        det.bbox = bbox_orig;
        if (best_class < static_cast<int>(class_names.size())) {
            det.class_name = class_names[best_class];
        }

        detections.push_back(det);
    }

    std::vector<int> keep = nms(detections, iou_threshold);

    std::vector<Detection> result;
    result.reserve(keep.size());
    for (int idx : keep) {
        result.push_back(detections[idx]);
    }

    std::sort(result.begin(), result.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    return result;
}

} // namespace yolo
