important files:

1. demo_rwkv.py
2. tools/onnx_RWKV_in_150_lines.py
3. llama/memory_pool.py


Step1:

cd tools

```
python3 onnx_RWKV_in_150_lines.py
```

to reuse it and export onnx models, remove models folder
set:
CONVERT_FLOAT16 = False
DUMP_INPUT = True
SAVE_ONLY = True

Step2:

```
bash model_to_oe.sh
bash model_to_oe2.sh
```

python3 demo_rwkv.py (revise onnxdir param and VERIFY_HB_ONNX)

Step3:

maybe: cd bpu_convert_yaml, bash create_yaml.sh

bash create_calib_dir.sh

bash makertbin.sh