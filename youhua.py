import onnx
from onnxoptimizer import optimize

# 加载ONNX模型
try:
    model = onnx.load('best.onnx')
except Exception as e:
    print(f"加载模型时出错: {e}")
    exit(1)

# 优化模型
try:
    passes = ["fuse_bn_into_conv"]
    model_optimized = optimize(model, passes)
except Exception as e:
    print(f"优化模型时出错: {e}")
    exit(1)

# 保存优化后的模型
try:
    onnx.save(model_optimized, 'model_optimized.onnx')
    print("优化后的模型已保存为 model_optimized.onnx")
except Exception as e:
    print(f"保存模型时出错: {e}")


