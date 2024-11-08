import onnx
from onnxsim import simplify

# 加载ONNX模型
model = onnx.load('model_optimized.onnx')

# 剪枝模型权重
model_simplified, _ = simplify(model)  # 提取简化后的模型

# 保存剪枝后的模型
onnx.save(model_simplified, 'model_pruned.onnx')

