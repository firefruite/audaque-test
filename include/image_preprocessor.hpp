#pragma once

#include "types.hpp"
#include <opencv2/opencv.hpp>

namespace yolo {

class ImagePreprocessor {
public:
    static constexpr int kInputSize = 640;
    static constexpr int kPadValue = 114;

    ImagePreprocessor() = default;

    PreprocessResult preprocess(const cv::Mat& image);

    static void computeLetterboxParams(int orig_w, int orig_h,
                                       float& scale, float& pad_w, float& pad_h);

    static BBox scaleBboxToOriginal(const BBox& bbox_640, const PreprocessResult& pp_result);

    static cv::Mat decodeFromBuffer(const std::vector<unsigned char>& buffer);
};

} // namespace yolo
