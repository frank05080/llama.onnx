#!/bin/bash

# Define base source directory
base_src_dir="/home/ros/share_dir/gitrepos/llama.onnx/data"

# Define the calibration data directory
calib_dir="${base_src_dir}/calib_data"

### Create head calib dir
dst_dir="input_calib_head_x"
mkdir -p "${calib_dir}/${dst_dir}"

# Initialize index for head_input.bin files
head_index=0

# Loop through all ptdump_inputs directories and copy head_input.bin if it exists
for src_dir in "${base_src_dir}"/ptdumped_inputs*; do
  file0="${src_dir}/head_input.bin"
  if [ -e "$file0" ]; then
    cp "$file0" "${calib_dir}/${dst_dir}/head_input_${head_index}.bin"
    head_index=$((head_index + 1))
  fi
done

# Initialize index for mixing input files
mixing_index_0=0
mixing_index_1=0
### Create mixings calib dir
# Loop through all files in each ptdump_inputs directory
for src_dir in "${base_src_dir}"/ptdumped_inputs*; do
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
      cp "$file_0" "${calib_dir}/${dst_dir_0}/mixing_${i}_input_0_${mixing_index_0}.bin"
      mixing_index_0=$((mixing_index_0 + 1))
    fi

    # Check for input_1 files and move them
    file_1="${src_dir}/mixing_${i}_input_1.bin"
    if [ -e "$file_1" ]; then
      cp "$file_1" "${calib_dir}/${dst_dir_1}/mixing_${i}_input_1_${mixing_index_1}.bin"
      mixing_index_1=$((mixing_index_1 + 1))
    fi
  done
done
