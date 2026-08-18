#include "image_preprocessor.hpp"

#include <algorithm>
#include <stdexcept>

namespace yolo {

void ImagePreprocessor::computeLetterboxParams(int orig_w, int orig_h,
                                                float& scale, float& pad_w, float& pad_h) {
    scale = static_cast<float>(std::min(kInputSize * 1.0 / orig_w, kInputSize * 1.0 / orig_h));
    int new_w = static_cast<int>(std::round(orig_w * scale));
    int new_h = static_cast<int>(std::round(orig_h * scale));
    pad_w = (kInputSize - new_w) / 2.0f;
    pad_h = (kInputSize - new_h) / 2.0f;
}

PreprocessResult ImagePreprocessor::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("Empty image provided for preprocessing");
    }

    PreprocessResult result;
    result.original_width = image.cols;
    result.original_height = image.rows;

    computeLetterboxParams(image.cols, image.rows, result.scale, result.pad_w, result.pad_h);

    int new_w = static_cast<int>(std::round(image.cols * result.scale));
    int new_h = static_cast<int>(std::round(image.rows * result.scale));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded,
                       static_cast<int>(std::round(result.pad_h)),
                       kInputSize - new_h - static_cast<int>(std::round(result.pad_h)),
                       static_cast<int>(std::round(result.pad_w)),
                       kInputSize - new_w - static_cast<int>(std::round(result.pad_w)),
                       cv::BORDER_CONSTANT,
                       cv::Scalar(kPadValue, kPadValue, kPadValue));

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    result.tensor.resize(1 * 3 * kInputSize * kInputSize);
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    for (int c = 0; c < 3; ++c) {
        std::copy(channels[c].begin<float>(), channels[c].end<float>(),
                  result.tensor.begin() + c * kInputSize * kInputSize);
    }

    return result;
}

BBox ImagePreprocessor::scaleBboxToOriginal(const BBox& bbox_640, const PreprocessResult& pp_result) {
    BBox result;
    result.x = (bbox_640.x - pp_result.pad_w) / pp_result.scale;
    result.y = (bbox_640.y - pp_result.pad_h) / pp_result.scale;
    result.width = bbox_640.width / pp_result.scale;
    result.height = bbox_640.height / pp_result.scale;
    return result;
}

cv::Mat ImagePreprocessor::decodeFromBuffer(const std::vector<unsigned char>& buffer) {
    cv::Mat mat = cv::imdecode(buffer, cv::IMREAD_COLOR);
    return mat;
}

} // namespace yolo
