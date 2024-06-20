import numpy as np
import torch

# Define the path to the .bin file
bin_filepath = '/home/ros/share_dir/gitrepos/llama.onnx/tools/inputs/embed_input.bin'

# Define the shape and dtype of the input tensor
# input_shape = (1, 3, 224, 224)  # Replace with the actual shape of your tensor
input_dtype = np.int32  # Replace with the actual dtype of your tensor if different

# Read the binary file into a numpy array
np_input = np.fromfile(bin_filepath, dtype=input_dtype)#.reshape(input_shape)

print(np_input)

# Convert the numpy array to a PyTorch tensor
tensor_input = torch.tensor(np_input)

print(tensor_input)
