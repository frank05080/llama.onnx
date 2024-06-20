import numpy as np
import torch

bin_filepath = '/home/ros/share_dir/gitrepos/llama.onnx/tools/inputs/embed_input.bin'
# input_shape = (1, 3, 224, 224)  # Replace with the actual shape of your tensor
input_dtype = np.int32  # Replace with the actual dtype of your tensor if different
# Read the binary file into a numpy array
np_input = np.fromfile(bin_filepath, dtype=input_dtype)#.reshape(input_shape)
print(np_input)
# Convert the numpy array to a PyTorch tensor
tensor_input = torch.tensor(np_input)
print(tensor_input)
print()


bin_filepath = '/home/ros/share_dir/gitrepos/llama.onnx/tools/inputs/head_input.bin'
# input_shape = (1, 3, 224, 224)  # Replace with the actual shape of your tensor
input_dtype = np.float32  # Replace with the actual dtype of your tensor if different
# Read the binary file into a numpy array
np_input = np.fromfile(bin_filepath, dtype=input_dtype)#.reshape(input_shape)
print(np_input.shape)
# Convert the numpy array to a PyTorch tensor
tensor_input = torch.tensor(np_input)
print(tensor_input.shape)
print(tensor_input.dtype)
print()


