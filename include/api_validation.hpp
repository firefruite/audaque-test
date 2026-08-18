#pragma once

#include <string>
#include <utility>

namespace yolo {

struct ParsedThreshold {
    float value;
    bool valid;
};

ParsedThreshold parseConfidence(const std::string& raw);
ParsedThreshold parseIou(const std::string& raw);

bool validateFileCount(size_t count);
bool isValidImageExtension(const std::string& filename);
bool isValidFileSize(size_t size);

constexpr size_t kMaxFileSize = 10 * 1024 * 1024;
constexpr size_t kMaxFiles = 5;
constexpr size_t kMinFiles = 1;

inline const std::string& defaultConfidenceStr() {
    static const std::string s = "0.25";
    return s;
}

inline const std::string& defaultIouStr() {
    static const std::string s = "0.45";
    return s;
}

} // namespace yolo
