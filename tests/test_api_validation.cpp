#include "test_utils.hpp"
#include "http_server.hpp"
#include "api_validation.hpp"
#include "detector.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace yolo;

static std::vector<std::string> loadLabels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream f(path);
    if (f.is_open()) {
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            labels.push_back(line);
        }
    }
    if (labels.empty()) labels.resize(80, "unknown");
    return labels;
}

static std::string makeFakeJpeg(int size_bytes) {
    // Minimal valid JPEG header + padding
    std::string data;
    data.reserve(size_bytes);
    data.push_back(static_cast<char>(0xFF));
    data.push_back(static_cast<char>(0xD8));
    data.push_back(static_cast<char>(0xFF));
    data.push_back(static_cast<char>(0xE0));
    data.append(size_bytes - 4, static_cast<char>(0x00));
    return data;
}

int main() {
    std::cout << "=== test_api_validation ===" << std::endl;
    int failures = 0;

    auto labels = loadLabels("assets/coco80.txt");
    OnnxDetector detector;
    std::string model_path = "models/yolo11n.onnx";

    bool model_exists = std::ifstream(model_path).good();

    std::cout << "  Model exists: " << (model_exists ? "yes" : "no") << std::endl;

    bool model_loaded = false;
    if (model_exists) {
        try {
            detector.loadModel(model_path);
            model_loaded = detector.isLoaded();
        } catch (...) {
            model_loaded = false;
        }
    }

    HttpServer server(detector, labels);

    // --- /health endpoint (unloaded model) ---
    {
        auto [code, j] = server.handleHealth();
        EXPECT_EQ(code, 200);
        EXPECT_EQ(j.value("model_loaded", true), false);
        EXPECT_EQ(j.value("model_name", ""), "yolo11n");
        EXPECT_EQ(j.value("status", ""), "ok");
        std::cout << "  [OK] /health returns 200, model_loaded=false when model not loaded" << std::endl;
    }

    // --- handleBatchInfer returns 503 when model not loaded ---
    {
        httplib::MultipartFormDataMap files;
        files.emplace("images", httplib::MultipartFormData{
            "images", makeFakeJpeg(100), "a.jpg", "image/jpeg"
        });
        auto [code, j] = server.handleBatchInfer(files, "", "");
        EXPECT_EQ(code, 503);
        EXPECT_EQ(j.value("model_loaded", true), false);
        std::cout << "  [OK] /v1/infer/batch returns 503 when model not loaded" << std::endl;
    }

    // --- Validation functions (no model needed) ---
    {
        EXPECT_TRUE(isValidImageExtension("test.jpg"));
        EXPECT_TRUE(isValidImageExtension("test.JPG"));
        EXPECT_TRUE(isValidImageExtension("test.jpeg"));
        EXPECT_TRUE(isValidImageExtension("test.JPEG"));
        EXPECT_TRUE(isValidImageExtension("test.png"));
        EXPECT_TRUE(isValidImageExtension("test.PNG"));
        EXPECT_FALSE(isValidImageExtension("test.txt"));
        EXPECT_FALSE(isValidImageExtension("test.pdf"));
        EXPECT_FALSE(isValidImageExtension("test.exe"));
        EXPECT_FALSE(isValidImageExtension("no_ext"));
        std::cout << "  [OK] isValidImageExtension" << std::endl;
    }

    {
        EXPECT_TRUE(validateFileCount(1));
        EXPECT_TRUE(validateFileCount(5));
        EXPECT_FALSE(validateFileCount(0));
        EXPECT_FALSE(validateFileCount(6));
        std::cout << "  [OK] validateFileCount" << std::endl;
    }

    {
        EXPECT_TRUE(isValidFileSize(1024));
        EXPECT_TRUE(isValidFileSize(10 * 1024 * 1024));
        EXPECT_FALSE(isValidFileSize(0));
        EXPECT_FALSE(isValidFileSize(10 * 1024 * 1024 + 1));
        EXPECT_FALSE(isValidFileSize(11 * 1024 * 1024));
        std::cout << "  [OK] isValidFileSize" << std::endl;
    }

    {
        auto c = parseConfidence("");
        EXPECT_TRUE(c.valid);
        EXPECT_NEAR(c.value, 0.25f, 0.001f);

        c = parseConfidence("0.5");
        EXPECT_TRUE(c.valid);
        EXPECT_NEAR(c.value, 0.5f, 0.001f);

        c = parseConfidence("-0.1");
        EXPECT_FALSE(c.valid);

        c = parseConfidence("1.1");
        EXPECT_FALSE(c.valid);

        c = parseConfidence("abc");
        EXPECT_FALSE(c.valid);

        c = parseConfidence("nan");
        EXPECT_FALSE(c.valid);

        std::cout << "  [OK] parseConfidence" << std::endl;
    }

    {
        auto iou = parseIou("");
        EXPECT_TRUE(iou.valid);
        EXPECT_NEAR(iou.value, 0.45f, 0.001f);

        iou = parseIou("0.7");
        EXPECT_TRUE(iou.valid);
        EXPECT_NEAR(iou.value, 0.7f, 0.001f);

        iou = parseIou("-0.5");
        EXPECT_FALSE(iou.valid);

        iou = parseIou("2.0");
        EXPECT_FALSE(iou.valid);

        std::cout << "  [OK] parseIou" << std::endl;
    }

    // --- Model-dependent tests (skip if model not available) ---
    if (model_loaded) {
        std::cout << "\n  [*] Model loaded, running model-dependent tests..." << std::endl;

        // --- /health endpoint (loaded model) ---
        {
            auto [code, j] = server.handleHealth();
            EXPECT_EQ(code, 200);
            EXPECT_EQ(j.value("model_loaded", false), true);
            std::cout << "  [OK] /health returns 200, model_loaded=true" << std::endl;
        }

        // --- No images → 400 ---
        {
            httplib::MultipartFormDataMap files;
            auto [code, j] = server.handleBatchInfer(files, "", "");
            EXPECT_EQ(code, 400);
            std::cout << "  [OK] No images → 400" << std::endl;
        }

        // --- Too many images (>5) → 400 ---
        {
            httplib::MultipartFormDataMap files;
            for (int i = 0; i < 6; ++i) {
                files.emplace("images", httplib::MultipartFormData{
                    "images", makeFakeJpeg(100), "img.jpg", "image/jpeg"
                });
            }
            auto [code, j] = server.handleBatchInfer(files, "", "");
            EXPECT_EQ(code, 400);
            std::cout << "  [OK] 6 images → 400" << std::endl;
        }

        // --- Non-image file → 400 ---
        {
            httplib::MultipartFormDataMap files;
            files.emplace("images", httplib::MultipartFormData{
                "images", makeFakeJpeg(100), "file.txt", "text/plain"
            });
            auto [code, j] = server.handleBatchInfer(files, "", "");
            EXPECT_EQ(code, 400);
            std::cout << "  [OK] Non-image file → 400" << std::endl;
        }

        // --- Oversized file → 400 ---
        {
            httplib::MultipartFormDataMap files;
            files.emplace("images", httplib::MultipartFormData{
                "images", makeFakeJpeg(11 * 1024 * 1024), "big.jpg", "image/jpeg"
            });
            auto [code, j] = server.handleBatchInfer(files, "", "");
            EXPECT_EQ(code, 400);
            std::cout << "  [OK] Oversized file → 400" << std::endl;
        }

        // --- Invalid confidence → 400 ---
        {
            httplib::MultipartFormDataMap files;
            files.emplace("images", httplib::MultipartFormData{
                "images", makeFakeJpeg(100), "a.jpg", "image/jpeg"
            });
            auto [code, j] = server.handleBatchInfer(files, "invalid", "");
            EXPECT_EQ(code, 400);
            std::cout << "  [OK] Invalid confidence → 400" << std::endl;
        }

        // --- Invalid iou → 400 ---
        {
            httplib::MultipartFormDataMap files;
            files.emplace("images", httplib::MultipartFormData{
                "images", makeFakeJpeg(100), "a.jpg", "image/jpeg"
            });
            auto [code, j] = server.handleBatchInfer(files, "0.25", "invalid");
            EXPECT_EQ(code, 400);
            std::cout << "  [OK] Invalid iou → 400" << std::endl;
        }

        // --- One invalid image in batch, others still process (200) ---
        // Note: fake JPEG will fail decoding, but batch should still return 200
        {
            httplib::MultipartFormDataMap files;
            files.emplace("images", httplib::MultipartFormData{
                "images", makeFakeJpeg(100), "bad.jpg", "image/jpeg"
            });
            files.emplace("images", httplib::MultipartFormData{
                "images", makeFakeJpeg(100), "good.jpg", "image/jpeg"
            });
            auto [code, j] = server.handleBatchInfer(files, "0.25", "0.45");
            EXPECT_EQ(code, 200);
            if (j.contains("results")) {
                EXPECT_EQ(j["results"].size(), 2);
            }
            std::cout << "  [OK] Batch with fake images returns 200 (decoding failures per-image)" << std::endl;
        }
    } else {
        std::cout << "\n  [*] Skipping model-dependent API tests (model not loaded)" << std::endl;
    }

    return test_util::summary();
}
