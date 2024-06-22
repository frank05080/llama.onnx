#!/bin/bash

# Define the base command and config path
base_command="hb_mapper makertbin --config"
config_path="/home/ros/share_dir/gitrepos/llama.onnx/bpu_convert_yaml"
src_dir="/home/ros/share_dir/gitrepos/llama.onnx/bpu_convert_yaml/model_convert_output"

config_file="${config_path}/rwkv_head_config.yaml"
full_command="${base_command} ${config_file} --model-type onnx"
echo "Running command: ${full_command}" 
$full_command
dst_dir="/home/ros/share_dir/gitrepos/llama.onnx/data/output_bin"
mkdir -p ${dst_dir}
if [ -d "$src_dir" ]; then
    mv "$src_dir" "${dst_dir}/model_convert_output_rwkv_head"
    echo "Renamed ${src_dir} to ${dst_dir}/model_convert_output_rwkv_head"
else
    echo "Directory ${src_dir} does not exist. Skipping rename."
fi
mv "/home/ros/share_dir/gitrepos/llama.onnx/hb_mapper_makertbin.log" "${dst_dir}/model_convert_output_rwkv_head"

###############################################

# Loop through mixing_0 to mixing_23
for i in {0..23}; do
  # Construct the config file name
  config_file="${config_path}/rwkv_mixing_${i}_config.yaml"
  # Construct and run the full command
  full_command="${base_command} ${config_file} --model-type onnx"
  echo "Running command: ${full_command}"
  $full_command

  # Rename the output folder
  if [ -d "$src_dir" ]; then
    mv "$src_dir" "${dst_dir}/model_convert_output_rwkv_mixing_${i}"
    echo "Renamed ${src_dir} to ${dst_dir}/model_convert_output_rwkv_mixing_${i}"
  else
    echo "Directory ${src_dir} does not exist. Skipping rename."
  fi

  mv "/home/ros/share_dir/gitrepos/llama.onnx/hb_mapper_makertbin.log" "${dst_dir}/model_convert_output_rwkv_mixing_${i}"

done
