## do hb_mapper checker to every onnx model, output each folder

#!/bin/bash
cd models

model_name="embed.onnx"
hb_mapper checker --model-type "onnx" --model "$model_name" --march "bernoulli2"
if [ -d ".hb_check" ]; then
  mv .hb_check "embed"
  echo "Renamed .hb_check to embed"
else
  echo "Error: .hb_check folder not found for embed"
fi

model_name="head.onnx"
hb_mapper checker --model-type "onnx" --model "$model_name" --march "bernoulli2"
if [ -d ".hb_check" ]; then
  mv .hb_check "head"
  echo "Renamed .hb_check to head"
else
  echo "Error: .hb_check folder not found for head"
fi

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
