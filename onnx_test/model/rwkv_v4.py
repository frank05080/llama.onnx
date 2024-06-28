
import os
from time import time

import numpy as np

from utils.memory_pool import OrtWrapper


class RWKV_RNN():
    def __init__(self, onnxdir: str, n_layer=24):
        self.total_infer_time = 0
        self.embed = OrtWrapper(os.path.join(onnxdir, 'embed.onnx'))
        self.head = OrtWrapper(os.path.join(onnxdir, 'head.onnx'))
        self.backbone = []
        for i in range(n_layer):
            self.backbone.append(OrtWrapper(os.path.join(onnxdir, 'mixing_{}.onnx'.format(i))))

    def forward(self, token, state):
        token = np.full((1), token, dtype=np.int32)
        start_time = time()
        x = self.embed.forward({'token': token})['output']
        end_time = time()
        self.total_infer_time += end_time - start_time
        for i, node in enumerate(self.backbone):
            state_in = state[5 * i:5 * i + 5]
            start_time = time()
            out = node.forward({'input': x.astype(np.float32), 'state_in': state_in})
            end_time = time()
            self.total_infer_time += end_time - start_time
            x = out['output']
            state[5 * i:5 * i + 5] = out['state_out']
        start_time = time()
        head_out = self.head.forward({'x': x.astype(np.float32)})['output']
        end_time = time()
        self.total_infer_time += end_time - start_time
        
        print("onnx infer time:", self.total_infer_time, "secs")
        self.total_infer_time = 0
        return head_out, state