#include "test_utils.hpp"
#include "postprocessor.hpp"
#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

using namespace yolo;

static PreprocessResult makePP(float scale, float pad_w, float pad_h, int ow, int oh) {
    PreprocessResult pp;
    pp.scale = scale;
    pp.pad_w = pad_w;
    pp.pad_h = pad_h;
    pp.original_width = ow;
    pp.original_height = oh;
    return pp;
}

int main() {
    std::cout << "=== test_postprocessor ===" << std::endl;

    std::vector<std::string> class_names;
    std::ifstream file("assets/coco80.txt");
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            class_names.push_back(line);
        }
    }
    if (class_names.empty()) {
        class_names.resize(80, "unknown");
    }

    Postprocessor pp;
    PreprocessResult pp_result = makePP(1.0f, 0.0f, 0.0f, 640, 640);

    // Model output format: [1, 84, 8400] = 4 bbox + 80 class scores
    // Channel layout:
    //   0-3  : bbox (cx, cy, w, h)
    //   4-83 : class scores (80 classes, sigmoid-activated, confidence = class score)
    int num_anchors = 8400;

    // --- Test 1: NMS same class high overlap removed ---
    {
        std::vector<float> output(84 * num_anchors, 0.0f);

        output[0 * num_anchors + 100] = 330.0f;  // cx
        output[1 * num_anchors + 100] = 330.0f;  // cy
        output[2 * num_anchors + 100] = 100.0f;  // w
        output[3 * num_anchors + 100] = 100.0f;  // h
        output[(4 + 0) * num_anchors + 100] = 0.95f; // class 0 (person)

        output[0 * num_anchors + 200] = 325.0f;
        output[1 * num_anchors + 200] = 325.0f;
        output[2 * num_anchors + 200] = 95.0f;
        output[3 * num_anchors + 200] = 95.0f;
        output[(4 + 0) * num_anchors + 200] = 0.90f;

        output[0 * num_anchors + 300] = 100.0f;
        output[1 * num_anchors + 300] = 100.0f;
        output[2 * num_anchors + 300] = 50.0f;
        output[3 * num_anchors + 300] = 50.0f;
        output[(4 + 2) * num_anchors + 300] = 0.8f; // class 2 (car)

        auto detections = pp.postprocess(output, pp_result, 0.25f, 0.45f, class_names);

        EXPECT_EQ(detections.size(), 2);
        if (detections.size() >= 1) {
            EXPECT_EQ(detections[0].class_id, 0);
            EXPECT_NEAR(detections[0].confidence, 0.95f, 0.001f);
        }
        if (detections.size() >= 2) {
            EXPECT_EQ(detections[1].class_id, 2);
            EXPECT_NEAR(detections[1].confidence, 0.8f, 0.001f);
        }
        std::cout << "  [OK] NMS same class high overlap removed (kept " << detections.size() << " detections)" << std::endl;
    }

    // --- Test 2: Different classes retained even with overlap ---
    {
        std::vector<float> output(84 * num_anchors, 0.0f);

        output[0 * num_anchors + 100] = 330.0f;
        output[1 * num_anchors + 100] = 330.0f;
        output[2 * num_anchors + 100] = 100.0f;
        output[3 * num_anchors + 100] = 100.0f;
        output[(4 + 0) * num_anchors + 100] = 0.9f;

        output[0 * num_anchors + 200] = 330.0f;
        output[1 * num_anchors + 200] = 330.0f;
        output[2 * num_anchors + 200] = 100.0f;
        output[3 * num_anchors + 200] = 100.0f;
        output[(4 + 2) * num_anchors + 200] = 0.9f;

        auto detections = pp.postprocess(output, pp_result, 0.25f, 0.45f, class_names);

        EXPECT_EQ(detections.size(), 2);
        std::cout << "  [OK] Different classes retained despite overlap (kept " << detections.size() << ")" << std::endl;
    }

    // --- Test 3: Confidence threshold filtering ---
    {
        std::vector<float> output(84 * num_anchors, 0.0f);

        output[0 * num_anchors + 100] = 330.0f;
        output[1 * num_anchors + 100] = 330.0f;
        output[2 * num_anchors + 100] = 100.0f;
        output[3 * num_anchors + 100] = 100.0f;
        output[(4 + 0) * num_anchors + 100] = 0.1f; // conf 0.1 < 0.25 threshold

        auto detections = pp.postprocess(output, pp_result, 0.25f, 0.45f, class_names);
        EXPECT_EQ(detections.size(), 0);
        std::cout << "  [OK] Low confidence detection filtered (>0.25)" << std::endl;
    }

    // --- Test 4: Results sorted by confidence descending ---
    {
        std::vector<float> output(84 * num_anchors, 0.0f);

        output[0 * num_anchors + 100] = 330.0f;
        output[1 * num_anchors + 100] = 330.0f;
        output[2 * num_anchors + 100] = 100.0f;
        output[3 * num_anchors + 100] = 100.0f;
        output[(4 + 0) * num_anchors + 100] = 0.9f;

        output[0 * num_anchors + 200] = 100.0f;
        output[1 * num_anchors + 200] = 100.0f;
        output[2 * num_anchors + 200] = 50.0f;
        output[3 * num_anchors + 200] = 50.0f;
        output[(4 + 2) * num_anchors + 200] = 0.8f;

        output[0 * num_anchors + 300] = 500.0f;
        output[1 * num_anchors + 300] = 500.0f;
        output[2 * num_anchors + 300] = 30.0f;
        output[3 * num_anchors + 300] = 30.0f;
        output[(4 + 16) * num_anchors + 300] = 0.95f;

        auto detections = pp.postprocess(output, pp_result, 0.25f, 0.45f, class_names);

        EXPECT_EQ(detections.size(), 3);
        if (detections.size() >= 3) {
            EXPECT_TRUE(detections[0].confidence >= detections[1].confidence);
            EXPECT_TRUE(detections[1].confidence >= detections[2].confidence);
        }
        std::cout << "  [OK] Results sorted by confidence descending" << std::endl;
    }

    // --- Test 5: Coordinate scaling to original image ---
    {
        std::vector<float> output(84 * num_anchors, 0.0f);

        PreprocessResult pp_res = makePP(2.0f, 0.0f, 80.0f, 320, 240);

        output[0 * num_anchors + 100] = 100.0f; // cx in 640 space
        output[1 * num_anchors + 100] = 80.0f;  // cy in 640 space
        output[2 * num_anchors + 100] = 60.0f;  // w in 640 space
        output[3 * num_anchors + 100] = 60.0f;  // h in 640 space
        output[(4 + 0) * num_anchors + 100] = 0.9f;

        auto detections = pp.postprocess(output, pp_res, 0.25f, 0.45f, class_names);

        EXPECT_EQ(detections.size(), 1);
        if (detections.size() >= 1) {
            // bbox_640: x=70, y=50, w=60, h=60
            // orig: x=(70-0)/2=35, y=(50-80)/2=-15, w=30, h=30
            EXPECT_NEAR(detections[0].bbox.x, 35.0f, 0.1f);
            EXPECT_NEAR(detections[0].bbox.y, -15.0f, 0.1f);
            EXPECT_NEAR(detections[0].bbox.width, 30.0f, 0.1f);
            EXPECT_NEAR(detections[0].bbox.height, 30.0f, 0.1f);
        }
        std::cout << "  [OK] Coordinate scaling to original image (x=35, y=-15, w=30, h=30)" << std::endl;
    }

    // --- Test 6: IoU calculation ---
    {
        BBox a{0, 0, 100, 100};
        BBox b{50, 50, 100, 100};
        float iou = Postprocessor::calculateIoU(a, b);
        EXPECT_NEAR(iou, 2500.0f / 17500.0f, 0.001f);
        std::cout << "  [OK] IoU calculation (expected ~0.143, got " << iou << ")" << std::endl;
    }

    // --- Test 7: No overlap IoU = 0 ---
    {
        BBox a{0, 0, 50, 50};
        BBox b{100, 100, 50, 50};
        float iou = Postprocessor::calculateIoU(a, b);
        EXPECT_NEAR(iou, 0.0f, 0.001f);
        std::cout << "  [OK] No overlap -> IoU = 0" << std::endl;
    }

    return test_util::summary();
}
