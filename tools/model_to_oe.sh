#!/bin/bash

# Loop from 0 to 23
for i in {0..23}
do
  # Format the model name
  model_name="mixing_${i}.onnx"
  
  # Run the hb_mapper checker command
  hb_mapper checker --model-type "onnx" --model "$model_name" --march "bernoulli2"
  
  # Rename the generated .hb_check folder to mixing_xx
  if [ -d ".hb_check" ]; then
    mv .hb_check "mixing_${i}"
    echo "Renamed .hb_check to mixing_${i}"
  else
    echo "Error: .hb_check folder not found for mixing_${i}"
  fi
done
