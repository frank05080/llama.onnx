### bpu supports onnx ir version <= 7
### bpu not support float16, convert to float32
# import types
import os
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=200)
from tokenizers import Tokenizer
from utils import sample_logits, OrtWrapper
from model.rwkv_v4 import RWKV_RNN
import argparse

VERIFY_HB_ONNX = True


def parse_args():
    parser = argparse.ArgumentParser(description='rwkv.onnx onnxruntime demo')
    parser.add_argument('--onnxdir', default="./pt2onnx_models", help='rwkv onnx model directory.')
    # parser.add_argument('--hb_onnxdir', default="/home/ros/share_dir/gitrepos/llama.onnx/data/hb_check_optimized_float_models", help='rwkv onnx model directory.')
    # parser.add_argument('--hb_onnxdir', default="/home/ros/share_dir/gitrepos/llama.onnx/data/hb_check_quantized_models", help='rwkv onnx model directory.') # compare with acc above
    parser.add_argument('--length', type=int, default=20, help='max output length.')
    parser.add_argument('--n_layer', type=int, default=24, help='layer number, use 24 by default.')
    parser.add_argument('--n_embd', type=int, default=1024, help='embedding length, use 1024 by default.')
    parser.add_argument('--tokenizer_path', type=str, default="../20B_tokenizer.json")
    args = parser.parse_args()
    return args

def main():
    ret = ""
    args = parse_args()
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    context = "\nWhat is Nvidia?"
    model = RWKV_RNN(args.onnxdir, n_layer=args.n_layer)
    state = np.zeros((args.n_layer * 5, args.n_embd), dtype=np.float32)
    print('\nPreprocessing context. {}'.format(context))
    for token in tokenizer.encode(context).ids:
        out, state = model.forward(token, state)
        # print('.', end="", flush=True)
    all_tokens = []
    out_last = 0
    for i in range(args.length):
        token = sample_logits(out.astype(np.float32))
        all_tokens += [token]
        tmp = tokenizer.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:  # only print when we have a valid utf-8 string
            ret += tmp
            # print(tmp, end="", flush=True)
            out_last = i + 1
        out, state = model.forward(token, state)
    print("result: ", ret)
    print('\n')

if __name__ == '__main__':
    main()

