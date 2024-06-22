import numpy as np
import os

# Path to the binary file
bin_filepath = '/root/rwkv/head_input.bin'

# Read the binary file
data = np.fromfile(bin_filepath, dtype=np.float32)
print(data)
print(data.shape)

# # If you know the shape of the original array, you can reshape it
# original_shape = (1024,)
# reshaped_data = data.reshape(original_shape)

# print("Data read from binary file:")
# print(reshaped_data)
# print("Shape of data:", reshaped_data.shape)
