important files:

1. demo_rwkv.py
2. tools/onnx_RWKV_in_150_lines.py
3. llama/memory_pool.py


Step1:

cd tools

```
python3 onnx_RWKV_in_150_lines.py (or use bash dump_multiple_inputs.sh to create many ptdumped_inputs)
```

to reuse it and export onnx models, remove models folder
set:
CONVERT_FLOAT16 = False
DUMP_INPUT = True
SAVE_ONLY = True

Step2:

```
bash model_to_oe.sh # to use hb_check - can be skipped
bash model_to_oe2.sh # gather hb_check_quantized_models in output_bin to use for demo_rwkv.py
```

python3 demo_rwkv.py (revise onnxdir param and VERIFY_HB_ONNX) - to use it after bash makertbin.sh

Step3:

bash create_calib_dir_multi.sh (bash create_calib_dir.sh for simple)

maybe: cd bpu_convert_yaml, bash create_yaml.sh + manually revise rwkv_head_config.yaml

bash makertbin.sh


-----------------------------------------------------

## dump model and inputs
python3 onnx_RWKV_in_150_lines.py
bash dump_multiple_inputs.sh

## gather calib data into calib dir

bash create_calib_dir_multi.sh

## get output bins and quantized models

bash makertbin.sh

## gathers quantized model

bash model_to_oe2.sh

## verify quantized model acc

python3 demo_rwkv.py