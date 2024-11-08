from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model = YOLO(r"D:\深度学习\ultralytics-main\runs\detect\best.pt")

# Export the model
model.export(format="openvino")  # creates 'yolov8n_openvino_model/'

# Load the exported OpenVINO model
ov_model = YOLO(r"D:\深度学习\ultralytics-main\runs\detect")

# Run inference
results = ov_model("https://ultralytics.com/images/bus.jpg")