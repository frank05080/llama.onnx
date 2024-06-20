### bpu supports onnx ir version <= 7
### bpu not support float16, convert to float32

# import types
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=200)
from tokenizers import Tokenizer
from llama import sample_logits, OrtWrapper, HBOrtWrapper
import argparse
import os

VERIFY_HB_ONNX = False


class RWKV_RNN():

    def __init__(self, onnxdir: str, n_layer=24):
        if VERIFY_HB_ONNX:
            self.embed = HBOrtWrapper(os.path.join(onnxdir, 'optimized_embed.onnx'))
            self.head = HBOrtWrapper(os.path.join(onnxdir, 'optimized_head.onnx'))
            self.backbone = []
            for i in range(n_layer):
                self.backbone.append(HBOrtWrapper(os.path.join(onnxdir, 'optimized_mixing_{}.onnx'.format(i))))
        else:
            self.embed = OrtWrapper(os.path.join(onnxdir, 'embed.onnx'))
            self.head = OrtWrapper(os.path.join(onnxdir, 'head.onnx'))
            self.backbone = []
            for i in range(n_layer):
                self.backbone.append(OrtWrapper(os.path.join(onnxdir, 'mixing_{}.onnx'.format(i))))

    def forward(self, token, state):
        token = np.full((1), token)#, dtype=np.int32)
        x = self.embed.forward({'token': token})['output'] # x has shape [1024], dtype: torch.float32

        for i, node in enumerate(self.backbone): # state has shape: [120, 1024]
            state_in = state[5 * i:5 * i + 5] # state_in has shape [5, 1024]
            # out = node.forward({'input': x.astype(np.float16), 'state_in': state_in})
            out = node.forward({'input': x.astype(np.float32), 'state_in': state_in})
            x = out['output']
            state[5 * i:5 * i + 5] = out['state_out']

        # return self.head.forward({'x': x.astype(np.float16)})['output'], state
        return self.head.forward({'x': x.astype(np.float32)})['output'], state


def parse_args():
    parser = argparse.ArgumentParser(description='rwkv.onnx onnxruntime demo')
    parser.add_argument('--onnxdir', default="/home/ros/share_dir/gitrepos/llama.onnx/tools/models", help='rwkv onnx model directory.')
    parser.add_argument('--length', type=int, default=100, help='max output length.')
    parser.add_argument('--n_layer', type=int, default=24, help='layer number, use 24 by default.')
    parser.add_argument('--n_embd', type=int, default=1024, help='embedding length, use 1024 by default.')
    args = parser.parse_args()
    return args


def main():
    tokenizer = Tokenizer.from_file("rwkv/20B_tokenizer.json")

    # context = "\nIn a shocking findin"
    context = "\nHorizonRobotics"
    # context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."

    args = parse_args()
    model = RWKV_RNN(args.onnxdir, n_layer=args.n_layer)

    # state = np.zeros((args.n_layer * 5, args.n_embd), dtype=np.float16)
    state = np.zeros((args.n_layer * 5, args.n_embd), dtype=np.float32)

    print('\nPreprocessing context. {}'.format(context))
    for token in tokenizer.encode(context).ids:
        # init_out, init_state = model.forward(token, init_state)
        out, state = model.forward(token, state)
        print('.', end="", flush=True)

    all_tokens = []
    out_last = 0
    # out, state = init_out, init_state
    for i in range(args.length):
        token = sample_logits(out.astype(np.float32))
        all_tokens += [token]
        tmp = tokenizer.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:  # only print when we have a valid utf-8 string
            print(tmp, end="", flush=True)
            out_last = i + 1
        # print("enter forward")
        out, state = model.forward(token, state)
    print('\n')


if __name__ == '__main__':
    main()
