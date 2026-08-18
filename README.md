# YOLO11n 批量目标检测 API 服务

## 选题说明

实现一个本机 CPU 运行的 HTTP API 服务。服务加载 YOLO11n ONNX 模型，对上传的 JPEG/PNG 图片批量执行目标检测，并返回类别、置信度、检测框和耗时。

## 技术栈

- C++17
- CMake 3.20+
- ONNX Runtime C/C++ API
- OpenCV
- cpp-httplib
- nlohmann/json

## 提示词产生方法

项目要求由 `project.md` 规范定义。每个开发轮次的实现内容都通过 AGENTS.md 的约定，以对话形式驱动。每轮交互的输入指令、代码 diff、commit hash、修改时间会被记录到 `records.jsonl`。

## JSONL 文件生成方法

在每次与 Kilo Code 的交互中，系统会：
1. 在 `records.jsonl` 中追加一条 JSON 记录，包含本轮 `prompt_content`、`modify_diff`、`commit_hash`、`modify_time`、`agent_type`、`dev_language`。
2. 每轮代码修改对应一次 Git commit，commit hash 存入 JSONL。
3. 不创建、修改或伪造 JSONL 过程记录；只完成代码、测试、Dockerfile 和文档。

## 构建说明

### 1. 安装 ONNX Runtime (macOS ARM64)

```bash
./scripts/setup-onnxruntime-macos-arm64.sh
export ONNXRUNTIME_ROOT="$HOME/.local/onnxruntime/onnxruntime-osx-arm64-1.20.1"
```

### 2. 准备模型和类别文件

- 将 `yolo11n.onnx` 放到 `models/` 目录
- 将 `coco80.txt`（80 行 COCO 类别名）放到 `assets/` 目录

### 3. 构建

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
ctest --output-on-failure
```

### 运行服务

```bash
./build/yolo_server --model models/yolo11n.onnx --labels assets/coco80.txt --port 8080
```

### 健康检查

```bash
curl -s http://localhost:8080/health | python3 -m json.tool
```

预期输出：

```json
{
  "status": "ok",
  "model_name": "yolo11n",
  "model_loaded": true
}
```

## API 接口

### GET /health

```json
{
  "status": "ok",
  "model_name": "yolo11n",
  "model_loaded": true
}
```

### POST /v1/infer/batch

请求类型：`multipart/form-data`，图片字段名 `images`，一次 1-5 张图片。

可选查询参数：
- `confidence`：置信度阈值，默认 `0.25`，范围 `[0, 1]`
- `iou`：NMS IoU 阈值，默认 `0.45`，范围 `[0, 1]`

示例请求（使用项目中 `samples/bus.jpg`）：

```bash
curl -s -X POST -F "images=@samples/bus.jpg" "http://localhost:8080/v1/infer/batch?confidence=0.25&iou=0.45" | python3 -m json.tool
```

响应示例（`samples/bus.jpg` 实际推理结果）：

```json
{
  "model_name": "yolo11n",
  "confidence_threshold": 0.25,
  "iou_threshold": 0.45,
  "results": [
    {
      "filename": "bus.jpg",
      "status": "success",
      "inference_ms": 82.5,
      "detections": [
        {
          "class_id": 5,
          "class_name": "bus",
          "confidence": 0.9392,
          "bbox": {
            "x": 11.92,
            "y": 228.39,
            "width": 787.31,
            "height": 506.82
          }
        },
        {
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.9020,
          "bbox": {
            "x": 48.59,
            "y": 397.96,
            "width": 194.62,
            "height": 506.58
          }
        },
        {
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.8493,
          "bbox": {
            "x": 670.56,
            "y": 392.59,
            "width": 139.45,
            "height": 487.03
          }
        },
        {
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.8328,
          "bbox": {
            "x": 223.12,
            "y": 405.58,
            "width": 122.11,
            "height": 454.14
          }
        },
        {
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.3993,
          "bbox": {
            "x": -0.06,
            "y": 550.19,
            "width": 66.06,
            "height": 321.59
          }
        }
      ]
    }
  ]
}
```

## 过程中遇到的问题和解决方法

1. **模型输出坐标理解**：YOLOv11 ONNX 模型输出为 `[1, 84, 8400]`，其中 bbox 值已是 640x640 像素坐标，需通过 letterbox 参数还原回原图。
2. **OpenCV 与 nlohmann/json 安装**：通过 conda 环境提供，CMakeLists 中通过 `CMAKE_PREFIX_PATH` 和 `find_package` 自动发现。
3. **cpp-httplib 依赖**：通过下载单文件 `httplib.h` 到系统 include 目录，确保 `#include <httplib.h>` 可用。

## 项目目录结构

- `include/`：头文件
- `src/`：源代码
- `tests/`：测试用例
- `models/`：ONNX 模型
- `assets/`：类别文件 `coco80.txt`
- `scripts/`：环境脚本
- `docs/`：文档

## License

See [LICENSE](LICENSE).
