important files:

1. demo_rwkv.py
2. tools/onnx_RWKV_in_150_lines.py
3. llama/memory_pool.py


Step1:

cd tools

```
python3 onnx_RWKV_in_150_lines.py
```

remove models folder
set:
CONVERT_FLOAT16 = False
DUMP_INPUT = True
SAVE_ONLY = True

Step2:

```
bash model_to_oe.sh
```
