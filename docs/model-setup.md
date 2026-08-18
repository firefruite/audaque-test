# YOLO11n and ONNX Runtime setup (macOS ARM64)

The application expects the model at `models/yolo11n.onnx`.  The model is a
large binary and is intentionally ignored by Git.  It must be an exported
YOLO11n detection model with an input shape of `1x3x640x640` and the standard
`[1,84,8400]` output.

Install the ARM64 ONNX Runtime release outside the repository:

```zsh
./scripts/setup-onnxruntime-macos-arm64.sh
export ONNXRUNTIME_ROOT="$HOME/.local/onnxruntime/onnxruntime-osx-arm64-1.20.1"
```

This machine also needs CMake 3.20 or newer. Install it with the local package
manager, then configure a build using the runtime discovered above:

```zsh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

`CMakeLists.txt` discovers `include/onnxruntime_cxx_api.h` and
`lib/libonnxruntime.dylib` under `ONNXRUNTIME_ROOT`. It does not use an x86_64
binary: use only the `onnxruntime-osx-arm64` release on Apple Silicon.
