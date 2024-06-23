## Gather hb_check models together

#!/bin/bash

# model_assign="optimized_float"
model_assign="quantized"

# Base directory where subfolders are located
# base_dir="/home/ros/share_dir/gitrepos/llama.onnx/data/hb_check_result" # hb_check needs to specify extra calib_dir param
base_dir="/home/ros/share_dir/gitrepos/llama.onnx/data/output_bin"

# Destination base directory
dest_base_dir="/home/ros/share_dir/gitrepos/llama.onnx/data"

goal_dir="${dest_base_dir}/hb_check_${model_assign}_models"
mkdir ${goal_dir}

# onnx_file="${base_dir}/embed_hb_check_results/${model_assign}_model.onnx"
# if [ -f "${onnx_file}" ]; then
#     cp "${onnx_file}" "${goal_dir}/embed.onnx"
# fi

# onnx_file="${base_dir}/head_hb_check_results/${model_assign}_model.onnx"
onnx_file="${base_dir}/model_convert_output_rwkv_head/rwkv_head_${model_assign}_model.onnx"
if [ -f "${onnx_file}" ]; then
    cp "${onnx_file}" "${goal_dir}/head.onnx"
fi

# Loop over all subfolders
for i in {0..23}; do
    # subfolder="${base_dir}/mixing_${i}_hb_check_results"
    subfolder="${base_dir}/model_convert_output_rwkv_mixing_${i}"
    
    for onnx_file in ${subfolder}/rwkv_mixing_${i}_${model_assign}_model.onnx; do
        if [ -f "${onnx_file}" ]; then
            cp "${onnx_file}" "${goal_dir}/mixing_${i}.onnx"
        fi
    done
done

echo "All .onnx files have been copied to their respective folders."
