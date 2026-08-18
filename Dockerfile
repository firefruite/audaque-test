FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY build/yolo_server /app/yolo_server

EXPOSE 8080

CMD ["/app/yolo_server", "--model", "/app/models/yolo11n.onnx", "--labels", "/app/assets/coco80.txt"]
