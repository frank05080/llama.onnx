### bpu supports onnx ir version <= 7
### bpu not support float16, convert to float32

# import types
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=200)
from tokenizers import Tokenizer
from llama import sample_logits, OrtWrapper, HBOrtWrapper
import argparse
import os


class RWKV_RNN():

    def __init__(self, onnxdir: str, hb_onnxdir: str, n_layer=24):
        self.embed = OrtWrapper(os.path.join(onnxdir, 'embed.onnx')) # hbort not support int32 input (only float and uint8)
        self.head = HBOrtWrapper(os.path.join(hb_onnxdir, 'head.onnx'))
        self.backbone = []
        for i in range(n_layer):
            self.backbone.append(HBOrtWrapper(os.path.join(hb_onnxdir, 'mixing_{}.onnx'.format(i))))
        self.save_inputs = False

    def forward(self, token, state):
        token = np.full((1), token, dtype=np.int32)
        x = self.embed.forward({'token': token})['output'] # x has shape [1024], dtype: torch.float32

        for i, node in enumerate(self.backbone): # state has shape: [120, 1024]
            state_in = state[5 * i:5 * i + 5] # state_in has shape [5, 1024], dtype: float32
            
            # save x and state_in as .bin
            if self.save_inputs:
                x.tofile("/home/ros/share_dir/gitrepos/llama.onnx/rwkv_bpu_board_infer/sample_inputs/inputs_0.bin")
                state_in.tofile("/home/ros/share_dir/gitrepos/llama.onnx/rwkv_bpu_board_infer/sample_inputs/inputs_1.bin")
                # np.save("/home/ros/share_dir/gitrepos/llama.onnx/rwkv_bpu_board_infer/sample_inputs/inputs_0.bin", x) # array([ 1.6081, -3.7124,  0.5397, ...,  0.5309,  1.1049, -0.5744]
                """
                array([[ 0.0871, -0.0505,  0.2889, ...,  0.0691, -0.0415, -0.1705],
                    [ 0.2204, -0.0237,  0.2476, ...,  0.0259, -0.0155, -0.056 ],
                    [ 4.3011, -0.0202,  0.5614, ...,  0.2988, -1.542 ,  0.0074],
                    [ 3.9995,  0.0234,  1.2233, ...,  1.    ,  1.    ,  1.    ],
                    [ 0.5663, -0.0203,  3.7538, ...,  5.3535,  4.6941, -1.0324]],
                """
                # np.save("/home/ros/share_dir/gitrepos/llama.onnx/rwkv_bpu_board_infer/sample_inputs/inputs_1.bin", state_in)
                break
            
            out = node.forward({'input': x.astype(np.float32), 'state_in': state_in})
            x = out['output']
            state[5 * i:5 * i + 5] = out['state_out']
            print("1111111")
        if self.save_inputs:
            import pdb
            pdb.set_trace()

        # return self.head.forward({'x': x.astype(np.float16)})['output'], state
        head_out = self.head.forward({'x': x.astype(np.float32)})['output'] # x is ndarray of shape (1024,), dtype: float32
        return head_out, state # head_out has shape: (50277,), state has shape (120, 1024)


def parse_args():
    parser = argparse.ArgumentParser(description='rwkv.onnx onnxruntime demo')
    parser.add_argument('--onnxdir', default="/home/ros/share_dir/gitrepos/llama.onnx/data/pt2onnx_models", help='rwkv onnx model directory.')
    # parser.add_argument('--hb_onnxdir', default="/home/ros/share_dir/gitrepos/llama.onnx/data/hb_check_optimized_float_models", help='rwkv onnx model directory.')
    parser.add_argument('--hb_onnxdir', default="/home/ros/share_dir/gitrepos/llama.onnx/data/hb_check_quantized_models", help='rwkv onnx model directory.') # compare with acc above
    parser.add_argument('--length', type=int, default=20, help='max output length.')
    parser.add_argument('--n_layer', type=int, default=24, help='layer number, use 24 by default.')
    parser.add_argument('--n_embd', type=int, default=1024, help='embedding length, use 1024 by default.')
    parser.add_argument("--save_inputs", type=bool, default=False)
    args = parser.parse_args()
    return args


def main():
    tokenizer = Tokenizer.from_file("rwkv/20B_tokenizer.json")

    # context = "\nIn a shocking findin"
    # context = "\nWhat is HorizonRobotics?"
    context = "\nWhat is Nvidia?"
    # context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."

    args = parse_args()
    model = RWKV_RNN(args.onnxdir, args.hb_onnxdir, n_layer=args.n_layer)

    # state = np.zeros((args.n_layer * 5, args.n_embd), dtype=np.float16)
    state = np.zeros((args.n_layer * 5, args.n_embd), dtype=np.float32)

    print('\nPreprocessing context. {}'.format(context))
    for token in tokenizer.encode(context).ids:
        # init_out, init_state = model.forward(token, init_state)
        out, state = model.forward(token, state)
        print('.', end="", flush=True)

    model.save_inputs = args.save_inputs
    all_tokens = []
    out_last = 0
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
