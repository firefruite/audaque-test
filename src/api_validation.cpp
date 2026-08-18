#include "api_validation.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace yolo {

static ParsedThreshold parseThreshold(const std::string& raw, float default_value) {
    ParsedThreshold result{default_value, true};

    if (raw.empty()) {
        result.value = default_value;
        return result;
    }

    try {
        size_t pos = 0;
        float val = std::stof(raw, &pos);
        if (pos != raw.size()) {
            result.valid = false;
            return result;
        }
        if (std::isnan(val) || std::isinf(val)) {
            result.valid = false;
            return result;
        }
        if (val < 0.0f || val > 1.0f) {
            result.valid = false;
            return result;
        }
        result.value = val;
    } catch (...) {
        result.valid = false;
    }
    return result;
}

ParsedThreshold parseConfidence(const std::string& raw) {
    return parseThreshold(raw, 0.25f);
}

ParsedThreshold parseIou(const std::string& raw) {
    return parseThreshold(raw, 0.45f);
}

bool validateFileCount(size_t count) {
    return count >= kMinFiles && count <= kMaxFiles;
}

static std::string lowerExt(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return out;
}

bool isValidImageExtension(const std::string& filename) {
    if (filename.empty()) return false;

    size_t dot = filename.rfind('.');
    if (dot == std::string::npos) return false;

    std::string ext = lowerExt(filename.substr(dot));
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png";
}

bool isValidFileSize(size_t size) {
    return size > 0 && size <= kMaxFileSize;
}

} // namespace yolo
