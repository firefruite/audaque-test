#include "http_server.hpp"
#include "detector.hpp"

#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static void loadClassNames(std::vector<std::string>& class_names, const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open class names file: " << path << std::endl;
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        class_names.push_back(line);
    }
}

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;

    std::string model_path = "models/yolo11n.onnx";
    std::string class_names_path = "assets/coco80.txt";
    int port = 8080;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--labels" && i + 1 < argc) {
            class_names_path = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [--model PATH] [--labels PATH] [--port N]\n";
            return 0;
        }
    }

    std::vector<std::string> class_names;
    loadClassNames(class_names, class_names_path);

    yolo::OnnxDetector detector;
    bool model_loaded = false;
    try {
        if (fs::exists(model_path)) {
            detector.loadModel(model_path);
            model_loaded = detector.isLoaded();
            if (!model_loaded) {
                std::cerr << "Failed to load model: " << model_path << std::endl;
            } else {
                std::cout << "Model loaded: " << detector.getModelName() << std::endl;
            }
        } else {
            std::cerr << "Model file not found: " << model_path << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
    }

    if (class_names.size() != 80) {
        std::cerr << "Warning: Expected 80 class names, got " << class_names.size() << std::endl;
    }

    yolo::HttpServer server(detector, class_names);

    std::cout << "Starting server on port " << port << std::endl;
    server.start(port);

    return 0;
}
