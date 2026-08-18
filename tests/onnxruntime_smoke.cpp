#include <onnxruntime_cxx_api.h>

int main() {
  const char* version = OrtGetApiBase()->GetVersionString();
  return version == nullptr ? 1 : 0;
}
