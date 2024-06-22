## do hb_mapper checker to every onnx model, output each folder

#!/bin/bash
cd /home/ros/share_dir/gitrepos/llama.onnx/data

if [ -d "hb_check_result" ]; then
  echo "hb_check_result folder exists"
else
  mkdir hb_check_result
fi

model_name="pt2onnx_models/embed.onnx"
hb_mapper checker --model-type "onnx" --model "$model_name" --march "bernoulli2"
if [ -d ".hb_check" ]; then
  mv .hb_check "hb_check_result/embed_hb_check_results"
  mv hb_mapper_checker.log "hb_check_result/embed_hb_check_results"
  echo "Renamed .hb_check to embed_hb_check_results"
else
  echo "Error: .hb_check folder not found for embed"
fi

model_name="pt2onnx_models/head.onnx"
hb_mapper checker --model-type "onnx" --model "$model_name" --march "bernoulli2"
if [ -d ".hb_check" ]; then
  mv .hb_check "hb_check_result/head_hb_check_results"
  mv hb_mapper_checker.log "hb_check_result/head_hb_check_results"
  echo "Renamed .hb_check to head_hb_check_results"
else
  echo "Error: .hb_check folder not found for head"
fi

# Loop from 0 to 23
for i in {0..23}
do
  # Format the model name
  model_name="pt2onnx_models/mixing_${i}.onnx"
  
  # Run the hb_mapper checker command
  hb_mapper checker --model-type "onnx" --model "$model_name" --march "bernoulli2"
  
  # Rename the generated .hb_check folder to mixing_xx
  if [ -d ".hb_check" ]; then
    mv .hb_check "hb_check_result/mixing_${i}_hb_check_results"
    mv hb_mapper_checker.log "hb_check_result/mixing_${i}_hb_check_results"
    echo "Renamed .hb_check to mixing_${i}_hb_check_results"
  else
    echo "Error: .hb_check folder not found for mixing_${i}"
  fi
done
