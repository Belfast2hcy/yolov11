from ultralytics import YOLO

# 加载模型
model = YOLO('D:/深度学习/ultralytics-main/yolo11m.pt')
# 如果要使用训练后的权重
model = YOLO(r"D:\深度学习\ultralytics-main\runs\detect\best.pt")

# 导出模型
model.export(format='onnx')

