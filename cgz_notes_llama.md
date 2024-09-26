在https://github.com/tpoisonooo/transformers/tree/add-convert这个分支中

src/transformers/models/llama/modeling_llama.py下，有torch.onnx.export的方法，可以帮助从源码中把llama模型导出onnx

安装这个版本的transformer:
```
cd transformers && python3 setup.py install
```
用pip3 show transformers，可以看到，对应的版本为4.28.0.dev0，对应的tokenizers版本为0.11.1


LlamaModel类是一个base model
LlamaForCausalLM就是在基础模型上加了一个nn.Linear


modelscope下载模型时，添加cache_dir参数，可以指定模型文件下载位置：
```Python
from modelscope import snapshot_download

model_dir = snapshot_download("AI-ModelScope/chinese-alpaca-2-7b", cache_dir="/home/ros/share_dir/gitrepos/llama.onnx/llama/llama_cache")
# /home/ros/.cache/modelscope/hub/AI-ModelScope/TinyLlama-1___1B-Chat-v1___0

print(model_dir)
```