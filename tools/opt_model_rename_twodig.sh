#!/bin/bash

# Loop from 01 to 23
for i in {1..23}
do
  # Format the original and new file names
  old_filename=$(printf "./models/optimized_mixing%02d.onnx" $i)
  new_filename=$(printf "./models/optimized_mixing_%02d.onnx" $i)

  # Check if the old file exists
  if [ -f "$old_filename" ]; then
    # Rename the file
    mv "$old_filename" "$new_filename"
    echo "Renamed $old_filename to $new_filename"
  else
    echo "File $old_filename not found"
  fi
done