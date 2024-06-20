import onnx

# Load the ONNX model
onnx_model_path = '/home/ros/share_dir/gitrepos/llama.onnx/tools/models/embed.onnx'
model = onnx.load(onnx_model_path)

# Get the IR version
ir_version = model.ir_version

print(f'The IR version of the model is: {ir_version}')

print(model.opset_import[0].version)
