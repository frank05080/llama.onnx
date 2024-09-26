from modelscope import snapshot_download

model_dir = snapshot_download("AI-ModelScope/chinese-alpaca-2-7b", cache_dir="/home/ros/share_dir/gitrepos/llama.onnx/llama/llama_cache")
# /home/ros/.cache/modelscope/hub/AI-ModelScope/TinyLlama-1___1B-Chat-v1___0

print(model_dir)