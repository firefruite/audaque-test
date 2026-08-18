项目名称：
YOLO11n 批量目标检测 API 服务

项目目标：
实现一个本机 CPU 运行的 HTTP API 服务。服务加载 YOLO11n ONNX 模型，对上传的 JPEG/PNG 图片批量执行目标检测，并返回类别、置信度、检测框和耗时。

技术栈：
- C++17
- CMake 3.20+
- ONNX Runtime C/C++ API
- OpenCV
- cpp-httplib：HTTP 服务
- nlohmann/json：JSON 序列化
- 模型文件：models/yolo11n.onnx
- 类别文件：assets/coco80.txt

类别文件生成方式：
- `coco80.txt` 以实际导出模型对应的 `yolo11n.pt` 的 `model.names` 为最终来源，保证类别索引与模型输出一致。
- 使用以下命令在项目根目录生成 `assets/coco80.txt`；文件必须包含 80 行，每行一个类别名称，行号对应 `class_id`。

```bash
python - <<'PY'
from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
labels = [model.names[index] for index in range(len(model.names))]

if len(labels) != 80:
    raise RuntimeError(f"Expected 80 COCO labels, got {len(labels)}")

Path("assets").mkdir(exist_ok=True)
Path("assets/coco80.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
print("Created assets/coco80.txt with 80 labels")
PY
```

- Ultralytics 官方 `coco.yaml` 的 `names` 字段可用于交叉核对，但不得改变 `model.names` 的顺序。

运行环境：
- macOS ARM64
- CPU 推理，不实现 GPU 推理
- ONNXRUNTIME_ROOT 环境变量指向 ONNX Runtime 的安装目录
- OpenCV 通过 CMake 的 find_package(OpenCV REQUIRED) 查找

必须实现的接口：

1. GET /health

返回服务状态、模型名称、模型是否成功加载。例如：

{
  "status": "ok",
  "model_name": "yolo11n",
  "model_loaded": true
}

模型未加载时仍返回可读状态，但推理接口应拒绝请求。

2. POST /v1/infer/batch

请求类型为 multipart/form-data。
图片字段名称固定为 images。
一次请求必须上传 1 至 5 张图片。
只允许 .jpg、.jpeg、.png。
单张文件最大 10 MB。

支持两个可选查询参数：
- confidence：置信度阈值，默认 0.25，范围 0 到 1
- iou：NMS IoU 阈值，默认 0.45，范围 0 到 1

响应示例：

{
  "model_name": "yolo11n",
  "confidence_threshold": 0.25,
  "iou_threshold": 0.45,
  "results": [
    {
      "filename": "street.jpg",
      "status": "success",
      "inference_ms": 38.5,
      "detections": [
        {
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.93,
          "bbox": {
            "x": 120.5,
            "y": 88.2,
            "width": 95.1,
            "height": 240.7
          }
        }
      ]
    },
    {
      "filename": "bad.png",
      "status": "error",
      "error": "Unable to decode image",
      "detections": []
    }
  ]
}

行为要求：
- 请求整体格式不合法、没有图片、图片超过 5 张、阈值非法时返回 HTTP 400。
- 模型未加载时返回 HTTP 503。
- 同一批中单张图片解码或推理失败时，返回 HTTP 200，并只在该图片结果中记录错误；其他图片继续处理。
- 检测框坐标必须还原到原图坐标系。
- 返回的检测按置信度从高到低排列。

模型推理流程：
1. 用 OpenCV 解码图片。
2. 使用 letterbox 方式等比例缩放并填充到 640 x 640。
3. 将 BGR 图像转换为 RGB，归一化到 0 到 1，并转换为 float32 的 NCHW 张量。
4. 使用 ONNX Runtime 执行 YOLO11n ONNX 模型推理。
5. 支持 YOLO11n 标准检测输出 [1, 84, 8400]；若模型输出不符合预期，返回清晰错误。
6. 解析中心点坐标、宽高和各类别分数。
7. 按 confidence 阈值过滤。
8. 按类别执行 NMS，使用 iou 阈值去除重叠框。
9. 将坐标从 640 x 640 预处理图还原到原图。

项目目录：

- CMakeLists.txt
- include/
  - detector.hpp
  - types.hpp
  - image_preprocessor.hpp
  - postprocessor.hpp
- src/
  - main.cpp
  - http_server.cpp
  - onnx_detector.cpp
  - image_preprocessor.cpp
  - postprocessor.cpp
- tests/
  - test_preprocessor.cpp
  - test_postprocessor.cpp
  - test_api_validation.cpp
- models/
  - .gitkeep
- assets/
  - coco80.txt
- samples/
  - .gitkeep
- docs/
  - model-setup.md
- Dockerfile
- README.md
- .gitignore

必须编写的测试：
- `/health` 在模型加载成功与失败时的响应。
- 无图片、超过 5 张图片、非图片文件、超大文件请求。
- `confidence` 和 `iou` 的非法参数。
- letterbox 后的缩放比例、填充值和坐标还原。
- NMS：同类别高重叠框被去除，不同类别框保留。
- 一批图片中有一张非法图片时，其余图片仍正常返回。

实现约束：
- 不实现模型训练、GPU、数据库、前端、鉴权、异步队列、消息队列、文件持久化。
- 不下载模型或依赖到仓库中；模型和原生运行时通过文档说明由使用者准备。
- 代码应分层：HTTP 层、推理层、图像预处理、后处理/NMS 分离。
- 所有错误必须是结构化、可读的 JSON 错误。
- 不创建、修改或伪造 JSONL 过程记录；只完成代码、测试、Dockerfile 和文档。
