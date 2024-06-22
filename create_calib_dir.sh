#!/bin/bash

# Define source directory
src_dir="/home/ros/share_dir/gitrepos/llama.onnx/data/ptdumped_inputs"
calib_dir="/home/ros/share_dir/gitrepos/llama.onnx/data/calib_data"

### Create head calib dir
dst_dir="input_calib_head_x"
mkdir -p "${calib_dir}/${dst_dir}"
file0="${src_dir}/head_input.bin"
if [ -e "$file0" ]; then
  cp "$file0" "${calib_dir}/${dst_dir}/"
fi

### Create mixings calib dir
# Loop through all files in the source directory
for i in {0..23}; do
  # Define destination directories based on the current index
  dst_dir_0="input_calib_mixing_${i}_input"
  dst_dir_1="input_calib_mixing_${i}_state_in"

  # Create destination directories if they don't exist
  mkdir -p "${calib_dir}/${dst_dir_0}"
  mkdir -p "${calib_dir}/${dst_dir_1}"

  # Check for input_0 files and move them
  file_0="${src_dir}/mixing_${i}_input_0.bin"
  if [ -e "$file_0" ]; then
    cp "$file_0" "${calib_dir}/${dst_dir_0}"
  fi

  # Check for input_1 files and move them
  file_1="${src_dir}/mixing_${i}_input_1.bin"
  if [ -e "$file_1" ]; then
    cp "$file_1" "${calib_dir}/${dst_dir_1}/"
  fi
done
