import numpy as np
import torch
# 加载地平线依赖库
from horizon_tc_ui import HB_ONNXRuntime

# 准备模型运行的输入, 生成形状为 (1,) 的 int32 数据
# input_data = np.random.randint(low=0, high=100, size=(1,), dtype=np.int32)
# input_data = torch.full([1], 25555, dtype=torch.int32).numpy()
input_bin_file = "/home/ros/share_dir/gitrepos/llama.onnx/data/calib_data/input_calib_head_x/head_input_0.bin"
# input_bin_file = "/home/ros/share_dir/gitrepos/llama.onnx/data/ptdumped_inputs/head_input.bin"
data = np.fromfile(input_bin_file, dtype=np.float32)
print(data)
print(data.shape)

# 加载模型文件
# sess = HB_ONNXRuntime(model_file = "/home/ros/share_dir/gitrepos/llama.onnx/data/output_bin/model_convert_output_rwkv_head/rwkv_head_quantized_model.onnx")
# sess = HB_ONNXRuntime(model_file = "/home/ros/share_dir/gitrepos/llama.onnx/data/output_bin/model_convert_output_rwkv_head/rwkv_head_calibrated_model.onnx")
sess = HB_ONNXRuntime(model_file = "/home/ros/share_dir/gitrepos/llama.onnx/data/output_bin/model_convert_output_rwkv_head/rwkv_head_original_float_model.onnx")
# sess = HB_ONNXRuntime(model_file="/home/ros/share_dir/gitrepos/llama.onnx/data/pt2onnx_models/head.onnx")
# 获取模型输入&输出节点信息
input_names = sess.input_names
output_names = sess.output_names
print(output_names)
# 准备输入数据，这里我们假设此模型只有1个输入
input_info = {input_names[0]: data}
# 开始模型推理，推理的返回值是一个list，依次与output_names指定名称一一对应
output = sess.run_feature(output_names, input_info)
print(output[0].shape)
print(output[0])
print()
print()


# ############################################################
# input_bin_file = "/home/ros/share_dir/gitrepos/llama.onnx/data/calib_data/input_calib_mixing_0_input/mixing_0_input_0.bin"
# # input_bin_file = "/home/ros/share_dir/gitrepos/llama.onnx/data/ptdumped_inputs/head_input.bin"
# data1 = np.fromfile(input_bin_file, dtype=np.float32)
# print(data1)
# print(data1.shape)

# input_bin_file = "/home/ros/share_dir/gitrepos/llama.onnx/data/calib_data/input_calib_mixing_0_state_in/mixing_0_input_1.bin"
# # input_bin_file = "/home/ros/share_dir/gitrepos/llama.onnx/data/ptdumped_inputs/head_input.bin"
# data2 = np.fromfile(input_bin_file, dtype=np.float32)
# data2 = data2.reshape((5,1024))
# print(data2)
# print(data2.shape)

# # 加载模型文件
# sess = HB_ONNXRuntime(model_file = "/home/ros/share_dir/gitrepos/llama.onnx/data/output_bin/model_convert_output_rwkv_mixing_0/rwkv_mixing_0_quantized_model.onnx")
# # sess = HB_ONNXRuntime(model_file = "/home/ros/share_dir/gitrepos/llama.onnx/data/output_bin/model_convert_output_rwkv_head/rwkv_head_calibrated_model.onnx")
# # sess = HB_ONNXRuntime(model_file = "/home/ros/share_dir/gitrepos/llama.onnx/data/output_bin/model_convert_output_rwkv_head/rwkv_head_original_float_model.onnx")
# # sess = HB_ONNXRuntime(model_file="/home/ros/share_dir/gitrepos/llama.onnx/data/pt2onnx_models/head.onnx")
# # 获取模型输入&输出节点信息
# input_names = sess.input_names
# output_names = sess.output_names
# print(output_names)
# # 准备输入数据，这里我们假设此模型只有1个输入
# input_info = {input_names[0]: data1, input_names[1]: data2}
# # 开始模型推理，推理的返回值是一个list，依次与output_names指定名称一一对应
# output = sess.run_feature(output_names, input_info)
# print("output[0].shape: ", output[0].shape)
# print("output[0]: ", output[0])
# print("output[1].shape: ", output[1].shape)
# print("output[1]: ", output[1])
# print()
# print()

# input_bin_file = "/home/ros/share_dir/gitrepos/llama.onnx/data/calib_data/input_calib_mixing_1_input/mixing_1_input_0.bin"
# data1 = np.fromfile(input_bin_file, dtype=np.float32)
# print(data1)
# print(data1.shape)

# input_bin_file = "/home/ros/share_dir/gitrepos/llama.onnx/data/calib_data/input_calib_mixing_1_state_in/mixing_1_input_1.bin"
# data2 = np.fromfile(input_bin_file, dtype=np.float32)
# data2 = data2.reshape((5,1024))
# print(data2)
# print(data2.shape)