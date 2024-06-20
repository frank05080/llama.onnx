import numpy as np
import torch

LAYER_NUM = 24

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

for i in range(LAYER_NUM):
    print(i)
    bin_filepath = '/home/ros/share_dir/gitrepos/llama.onnx/tools/inputs/mixing_{}_input_0.bin'.format(i)
    # input_shape = (1, 3, 224, 224)  # Replace with the actual shape of your tensor
    input_dtype = np.float32  # Replace with the actual dtype of your tensor if different
    # Read the binary file into a numpy array
    np_input = np.fromfile(bin_filepath, dtype=input_dtype)#.reshape(input_shape)
    print(np_input.shape)
    assert(np_input.shape[0] == 1024)
    assert(len(np_input.shape) == 1)
    # Convert the numpy array to a PyTorch tensor
    tensor_input = torch.tensor(np_input)
    print(tensor_input.shape)
    print(tensor_input.dtype)
    print()

    bin_filepath = '/home/ros/share_dir/gitrepos/llama.onnx/tools/inputs/mixing_{}_input_1.bin'.format(i)
    input_shape = (5, 1024)  # Replace with the actual shape of your tensor
    input_dtype = np.float32  # Replace with the actual dtype of your tensor if different
    # Read the binary file into a numpy array
    np_input = np.fromfile(bin_filepath, dtype=input_dtype).reshape(input_shape)
    print(np_input.shape)
    assert(np_input.shape[0] == 5)
    assert(np_input.shape[1] == 1024)
    assert(len(np_input.shape) == 2)
    # Convert the numpy array to a PyTorch tensor
    tensor_input = torch.tensor(np_input)
    print(tensor_input.shape)
    print(tensor_input.dtype)
    print()


