from openvino.runtime import Core, serialize

ie = Core()
onnx_model_path = r"D:\深度学习\ultralytics-main\model_pruned.onnx"
model_onnx = ie.read_model(model=onnx_model_path)

# 如果需要编译模型，可以取消注释以下行
# compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")

# 序列化模型
serialize(
    model=model_onnx,
    xml_path=r"D:\深度学习\ultralytics-main\exported_onnx_model.xml",
    bin_path=r"D:\深度学习\ultralytics-main\exported_onnx_model.bin",
    version="UNSPECIFIED"
)