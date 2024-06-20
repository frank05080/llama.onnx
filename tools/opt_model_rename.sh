#!/bin/bash

# Loop from 1 to 23
for i in {1..9}
do
  # Format the original and new file names with leading zeros for single-digit numbers
  old_filename=$(printf "./models/optimized_mixing%d.onnx" $i)
  new_filename=$(printf "./models/optimized_mixing_%d.onnx" $i)

  # Check if the old file exists
  if [ -f "$old_filename" ]; then
    # Rename the file
    mv "$old_filename" "$new_filename"
    echo "Renamed $old_filename to $new_filename"
  else
    echo "File $old_filename not found"
  fi
done