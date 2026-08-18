#pragma once

#include "types.hpp"
#include "detector.hpp"
#include "image_preprocessor.hpp"
#include "postprocessor.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace yolo {

class HttpServer {
public:
    HttpServer(OnnxDetector& detector, const std::vector<std::string>& class_names);

    void start(int port);
    void stop();

    std::pair<int, nlohmann::json> handleHealth();

    std::pair<int, nlohmann::json> handleBatchInfer(
        const httplib::MultipartFormDataMap& files,
        const std::string& raw_confidence,
        const std::string& raw_iou
    );

private:
    ImageResult processImage(const std::string& content, const std::string& filename,
                             float confidence_threshold, float iou_threshold);

    nlohmann::json errorResponse(const std::string& message);

    OnnxDetector& detector_;
    ImagePreprocessor preprocessor_;
    Postprocessor postprocessor_;
    std::vector<std::string> class_names_;
    httplib::Server server_;
};

} // namespace yolo
