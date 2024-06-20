## move the optimized model out

#!/bin/bash

# Loop from 0 to 23
for i in {0..23}
do
  # Format the folder name
  folder_name="mixing_${i}"
  
  # Check if the folder exists
  if [ -d "$folder_name" ]; then
    # Define the source file and target file names
    source_file="${folder_name}/optimized_float_model.onnx"
    target_file="optimized_mixing${i}.onnx"
    
    # Check if the source file exists
    if [ -f "$source_file" ]; then
      # Move and rename the file
      mv "$source_file" "$target_file"
      mv "$target_file" "/home/ros/share_dir/gitrepos/llama.onnx/tools/models"
      echo "Moved and renamed $source_file to $target_file"
    else
      echo "Error: $source_file not found in $folder_name"
    fi
  else
    echo "Error: Folder $folder_name not found"
  fi
done
