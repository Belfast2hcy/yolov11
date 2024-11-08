import onnx
import netron

# 加载ONNX模型
model = onnx.load('model_optimized.onnx')

# 可视化模型
netron.start('model_optimized.onnx')  # 使用模型文件的路径

