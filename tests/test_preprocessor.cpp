#include "test_utils.hpp"
#include "image_preprocessor.hpp"

#include <opencv2/opencv.hpp>
#include <cmath>

using namespace yolo;

static cv::Mat makeImage(int w, int h, const cv::Scalar& color) {
    cv::Mat img(h, w, CV_8UC3, color);
    return img;
}

int main() {
    std::cout << "=== test_preprocessor ===" << std::endl;

    ImagePreprocessor proc;

    // --- Letterbox scaling ratio ---

    // Test 1: Square image 640x640 → scale 1.0, no padding
    {
        cv::Mat img = makeImage(640, 640, cv::Scalar(100, 150, 200));
        auto result = proc.preprocess(img);
        EXPECT_NEAR(result.scale, 1.0f, 0.001f);
        EXPECT_NEAR(result.pad_w, 0.0f, 0.5f);
        EXPECT_NEAR(result.pad_h, 0.0f, 0.5f);
        EXPECT_EQ(result.original_width, 640);
        EXPECT_EQ(result.original_height, 640);
        EXPECT_EQ(result.tensor.size(), 3 * 640 * 640);
        std::cout << "  [OK] Square 640x640 → scale=1.0, no padding" << std::endl;
    }

    // Test 2: Wide image 1280x640 → scale 0.5, pad top/bottom
    {
        cv::Mat img = makeImage(1280, 640, cv::Scalar(100, 150, 200));
        auto result = proc.preprocess(img);
        EXPECT_NEAR(result.scale, 0.5f, 0.01f);
        EXPECT_NEAR(result.pad_w, 0.0f, 0.5f);
        EXPECT_NEAR(result.pad_h, 160.0f, 0.5f);
        std::cout << "  [OK] Wide 1280x640 → scale=0.5, pad_h=160" << std::endl;
    }

    // Test 3: Tall image 640x1280 → scale 0.5, pad left/right
    {
        cv::Mat img = makeImage(640, 1280, cv::Scalar(100, 150, 200));
        auto result = proc.preprocess(img);
        EXPECT_NEAR(result.scale, 0.5f, 0.01f);
        EXPECT_NEAR(result.pad_w, 160.0f, 0.5f);
        EXPECT_NEAR(result.pad_h, 0.0f, 0.5f);
        std::cout << "  [OK] Tall 640x1280 → scale=0.5, pad_w=160" << std::endl;
    }

    // Test 4: Small image 320x240 → scale 2.0, pad both
    {
        cv::Mat img = makeImage(320, 240, cv::Scalar(100, 150, 200));
        auto result = proc.preprocess(img);
        EXPECT_NEAR(result.scale, 2.0f, 0.01f);
        // new_w = 640, new_h = 480, pad_w = 0, pad_h = 80
        EXPECT_NEAR(result.pad_w, 0.0f, 0.5f);
        EXPECT_NEAR(result.pad_h, 80.0f, 0.5f);
        std::cout << "  [OK] Small 320x240 → scale=2.0, pad_h=80" << std::endl;
    }

    // --- Padding values ---

    // Test 5: Padding value should be ~114/255 = 0.447 in normalized space
    {
        cv::Mat img = makeImage(1280, 640, cv::Scalar(0, 0, 0)); // black image
        auto result = proc.preprocess(img);
        // In the padded region (top/bottom), all channels should be 114/255 ≈ 0.447
        // The tensor is NCHW, channel first
        int total_pixels = 640 * 640;
        int pad_pixel_count = static_cast<int>(160.0f * 640); // 160 rows of padding
        // Check a pixel in the padding region (top padding)
        // In NCHW: channel 0 is R, offset = 0*total_pixels + row*640 + col
        // Padding is top: rows 0..159 are padding (pad_h = 160)
        // At row 0, col 0: R channel
        int idx = 0 * total_pixels + 0 * 640 + 0;
        float r_val = result.tensor[idx];
        float expected = 114.0f / 255.0f;
        EXPECT_NEAR(r_val, expected, 0.001f);
        std::cout << "  [OK] Padding value = 114/255 ≈ 0.447" << std::endl;
    }

    // --- Coordinate restoration ---

    // Test 6: Coordinate restoration from 640x640 back to original
    {
        cv::Mat img = makeImage(1280, 640, cv::Scalar(0, 0, 0));
        auto result = proc.preprocess(img);
        // scale=0.5, pad_w=0, pad_h=160

        // A bbox at the center of 640x640: (320, 320) center
        // x = 320 - 50, y = 320 - 50, w = 100, h = 100
        BBox bbox_640;
        bbox_640.x = 270.0f;
        bbox_640.y = 270.0f;
        bbox_640.width = 100.0f;
        bbox_640.height = 100.0f;

        BBox orig = proc.scaleBboxToOriginal(bbox_640, result);
        // x_orig = (270 - 0) / 0.5 = 540
        // y_orig = (270 - 160) / 0.5 = 110 / 0.5 = 220
        // w_orig = 100 / 0.5 = 200
        // h_orig = 100 / 0.5 = 200
        EXPECT_NEAR(orig.x, 540.0f, 1.0f);
        EXPECT_NEAR(orig.y, 220.0f, 1.0f);
        EXPECT_NEAR(orig.width, 200.0f, 1.0f);
        EXPECT_NEAR(orig.height, 200.0f, 1.0f);
        std::cout << "  [OK] Coordinate restoration (1280x640, scale=0.5, pad_h=160)" << std::endl;
    }

    // Test 7: Coordinate restoration with wide padding (tall image)
    {
        cv::Mat img = makeImage(640, 1280, cv::Scalar(0, 0, 0));
        auto result = proc.preprocess(img);
        // scale=0.5, pad_w=160, pad_h=0

        BBox bbox_640;
        bbox_640.x = 170.0f;
        bbox_640.y = 340.0f;
        bbox_640.width = 100.0f;
        bbox_640.height = 100.0f;

        BBox orig = proc.scaleBboxToOriginal(bbox_640, result);
        // x_orig = (170 - 160) / 0.5 = 10 / 0.5 = 20
        // y_orig = (340 - 0) / 0.5 = 340 / 0.5 = 680
        // w_orig = 100 / 0.5 = 200
        // h_orig = 100 / 0.5 = 200
        EXPECT_NEAR(orig.x, 20.0f, 1.0f);
        EXPECT_NEAR(orig.y, 680.0f, 1.0f);
        EXPECT_NEAR(orig.width, 200.0f, 1.0f);
        EXPECT_NEAR(orig.height, 200.0f, 1.0f);
        std::cout << "  [OK] Coordinate restoration (640x1280, scale=0.5, pad_w=160)" << std::endl;
    }

    // Test 8: Tensor is in NCHW format, RGB order, normalized 0-1
    {
        cv::Mat img = cv::Mat(640, 640, CV_8UC3);
        cv::rectangle(img, cv::Rect(0, 0, 640, 640), cv::Scalar(255, 0, 0), -1); // Blue (BGR)
        auto result = proc.preprocess(img);
        // Blue in BGR → R=0, G=0, B=255 → normalized: R=0, G=0, B=1
        // After RGB conversion: R=0, G=0, B=1
        // Channel 0 (R) at pixel (0,0): 0/255 = 0.0
        int idx_r = 0 * 640 * 640 + 0 * 640 + 0;
        EXPECT_NEAR(result.tensor[idx_r], 0.0f, 0.001f);
        int idx_b = 2 * 640 * 640 + 0 * 640 + 0;
        EXPECT_NEAR(result.tensor[idx_b], 1.0f, 0.001f);
        std::cout << "  [OK] Tensor is NCHW RGB normalized 0-1" << std::endl;
    }

    return test_util::summary();
}
