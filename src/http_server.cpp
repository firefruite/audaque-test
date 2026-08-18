#include "http_server.hpp"
#include "api_validation.hpp"

#include <chrono>
#include <stdexcept>

namespace yolo {

HttpServer::HttpServer(OnnxDetector& detector, const std::vector<std::string>& class_names)
    : detector_(detector)
    , class_names_(class_names) {
}

nlohmann::json HttpServer::errorResponse(const std::string& message) {
    nlohmann::json j;
    j["error"] = message;
    return j;
}

std::pair<int, nlohmann::json> HttpServer::handleHealth() {
    nlohmann::json j;
    j["status"] = "ok";
    j["model_name"] = detector_.getModelName();
    j["model_loaded"] = detector_.isLoaded();
    return {200, j};
}

ImageResult HttpServer::processImage(
    const std::string& content,
    const std::string& filename,
    float confidence_threshold,
    float iou_threshold
) {
    ImageResult result;
    result.filename = filename;
    result.inference_ms = 0.0f;

    std::vector<unsigned char> buffer(content.begin(), content.end());
    cv::Mat image = preprocessor_.decodeFromBuffer(buffer);

    if (image.empty()) {
        result.status = "error";
        result.error = "Unable to decode image";
        return result;
    }

    PreprocessResult pp_result = preprocessor_.preprocess(image);

    auto start = std::chrono::steady_clock::now();
    std::vector<float> output = detector_.infer(pp_result.tensor);
    auto end = std::chrono::steady_clock::now();

    result.inference_ms = std::chrono::duration<float, std::milli>(end - start).count();

    try {
        std::vector<Detection> detections = postprocessor_.postprocess(
            output, pp_result, confidence_threshold, iou_threshold, class_names_);
        result.detections = std::move(detections);
        result.status = "success";
    } catch (const std::exception& e) {
        result.status = "error";
        result.error = std::string("Inference failed: ") + e.what();
    }

    return result;
}

std::pair<int, nlohmann::json> HttpServer::handleBatchInfer(
    const httplib::MultipartFormDataMap& files,
    const std::string& raw_confidence,
    const std::string& raw_iou
) {
    if (!detector_.isLoaded()) {
        nlohmann::json j;
        j["status"] = "error";
        j["model_name"] = detector_.getModelName();
        j["model_loaded"] = false;
        j["error"] = "Model not loaded";
        return {503, j};
    }

    ParsedThreshold conf = parseConfidence(raw_confidence);
    if (!conf.valid) {
        return {400, errorResponse("Invalid confidence parameter: must be a number between 0 and 1")};
    }

    ParsedThreshold iou = parseIou(raw_iou);
    if (!iou.valid) {
        return {400, errorResponse("Invalid iou parameter: must be a number between 0 and 1")};
    }

    std::vector<const httplib::MultipartFormData*> image_files;
    for (const auto& [name, file] : files) {
        if (name == "images") {
            image_files.push_back(&file);
        }
    }

    if (!validateFileCount(image_files.size())) {
        return {400, errorResponse("Invalid number of images: must upload between 1 and 5 images")};
    }

    for (const auto* file : image_files) {
        if (!isValidImageExtension(file->filename)) {
            return {400, errorResponse("Invalid file type for '" + file->filename + "': only .jpg, .jpeg, .png are allowed")};
        }
        if (!isValidFileSize(file->content.size())) {
            return {400, errorResponse("File too large for '" + file->filename + "': maximum size is 10 MB")};
        }
    }

    nlohmann::json j;
    j["model_name"] = detector_.getModelName();
    j["confidence_threshold"] = conf.value;
    j["iou_threshold"] = iou.value;
    nlohmann::json results = nlohmann::json::array();

    for (const auto* file : image_files) {
        ImageResult img_result = processImage(
            file->content, file->filename,
            conf.value, iou.value);

        nlohmann::json r;
        r["filename"] = img_result.filename;
        r["status"] = img_result.status;
        r["inference_ms"] = img_result.inference_ms;

        if (img_result.status == "error") {
            r["error"] = img_result.error;
        }

        nlohmann::json detections = nlohmann::json::array();
        for (const auto& d : img_result.detections) {
            nlohmann::json det;
            det["class_id"] = d.class_id;
            det["class_name"] = d.class_name;
            det["confidence"] = d.confidence;
            nlohmann::json bbox;
            bbox["x"] = d.bbox.x;
            bbox["y"] = d.bbox.y;
            bbox["width"] = d.bbox.width;
            bbox["height"] = d.bbox.height;
            det["bbox"] = bbox;
            detections.push_back(det);
        }
        r["detections"] = detections;
        results.push_back(r);
    }

    j["results"] = results;
    return {200, j};
}

void HttpServer::start(int port) {
    server_.Get("/health", [this](const httplib::Request& req, httplib::Response& res) {
        auto [code, j] = handleHealth();
        res.set_content(j.dump(2), "application/json");
        res.status = code;
    });

    server_.Post("/v1/infer/batch", [this](const httplib::Request& req, httplib::Response& res) {
        auto [code, j] = handleBatchInfer(req.files,
                                          req.has_param("confidence") ? req.get_param_value("confidence") : "",
                                          req.has_param("iou") ? req.get_param_value("iou") : "");
        res.set_content(j.dump(2), "application/json");
        res.status = code;
    });

    server_.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
        try {
            if (ep) std::rethrow_exception(ep);
            nlohmann::json j;
            j["error"] = "Internal server error";
            res.set_content(j.dump(2), "application/json");
            res.status = 500;
        } catch (const std::exception& e) {
            nlohmann::json j;
            j["error"] = std::string("Internal server error: ") + e.what();
            res.set_content(j.dump(2), "application/json");
            res.status = 500;
        }
    });

    server_.listen("0.0.0.0", port);
}

void HttpServer::stop() {
    server_.stop();
}

} // namespace yolo
