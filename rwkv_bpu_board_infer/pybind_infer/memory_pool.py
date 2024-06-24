from loguru import logger

from utils import singleton

import onnxruntime as ort

import numpy as np

import os

import sys



import psutil

import math



class OrtWrapper:

    def __init__(self, onnxfile: str):

        assert os.path.exists(onnxfile)

        self.onnxfile = onnxfile

        self.sess = ort.InferenceSession(onnxfile)

        self.inputs = self.sess.get_inputs()

        outputs = self.sess.get_outputs()

        self.output_names = [output.name for output in outputs]

        logger.debug('{} loaded'.format(onnxfile))



    def forward(self, _inputs: dict):

        assert len(self.inputs) == len(_inputs)

        output_tensors = self.sess.run(None, _inputs)



        assert len(output_tensors) == len(self.output_names)

        output = dict()

        for i, tensor in enumerate(output_tensors):

            output[self.output_names[i]] = tensor



        return output

    

    def __del__(self):

        logger.debug('{} unload'.format(self.onnxfile))

        



# use horizon_bpu conda env

import bpu_infer_lib



# def prepare_input_dict(input_names):

#     feed_dict = dict()

#     for input_name in input_names:

#         feed_dict[input_name] = data_prepare(input_name)



class HBOrtWrapper:
    # def forward(self, onnxfile: str, input0_bin: str, input1_bin: str, ind: int):
    def forward(self, onnxfile: str, input0_bin: str, input1_bin: str):
        # ptr1, ptr2 = bpu_infer_lib.mixing_infer(onnxfile, input0_bin, input1_bin, ind)
        ptr1, ptr2 = bpu_infer_lib.mixing_infer(onnxfile, input0_bin, input1_bin)
        array1 = [ptr1[i] for i in range(1024)]
        array2 = [ptr2[i] for i in range(1024*5)]
        out_arr = np.array(array1).astype(np.float32) # BUG 这里出来默认为float64，会出现精度问题
        out_arr2 = np.array(array2).reshape(5,1024).astype(np.float32)
        return {"output": out_arr, "state_out": out_arr2}
    
class HBHeadOrtWrapper:
    def forward(self, onnxfile: str, input0_bin: str):
        print("ready to head_infer")
        # ptr1 = bpu_infer_lib.head_infer(onnxfile, input0_bin, 24)
        ptr1 = bpu_infer_lib.head_infer(onnxfile, input0_bin)
        print("finish")
        array1 = [ptr1[i] for i in range(50277)]
        out_arr = np.array(array1).astype(np.float32) # BUG 这里出来默认为float64，会出现精度问题
        return {"output": out_arr}


@singleton
class MemoryPoolSimple:
    def __init__(self, maxGB):
        if maxGB < 0:
            raise Exception('maxGB must > 0, get {}'.format(maxGB))

        self.max_size = maxGB * 1024 * 1024 * 1024
        self.wait_map = {}
        self.active_map = {}

    def submit(self, key: str, onnx_filepath: str):
        if not os.path.exists(onnx_filepath):
            raise Exception('{} not exist!'.format(onnx_filepath))

        if key not in self.wait_map:
            self.wait_map[key] = {
                'onnx': onnx_filepath,
                'file_size': os.path.getsize(onnx_filepath)
            }

    def used(self):

        sum_size = 0

        biggest_k = None

        biggest_size = 0

        for k in self.active_map.keys():

            cur_size = self.wait_map[k]['file_size']

            sum_size += cur_size



            if biggest_k is None:

                biggest_k = k

                biggest_size = cur_size

                continue

            

            if cur_size > biggest_size:

                biggest_size = cur_size

                biggest_k = k

        

        return sum_size, biggest_k



    def check(self):

        sum_need = 0

        for k in self.wait_map.keys():

            sum_need = sum_need + self.wait_map[k]['file_size']

            

        sum_need /= (1024 * 1024 * 1024)

        

        total = psutil.virtual_memory().total / (1024 * 1024 * 1024)

        if total > 0 and total < sum_need:

            logger.warning('virtual_memory not enough, require {}, try `--poolsize {}`'.format(sum_need, math.floor(total)))





    def fetch(self, key: str):

        if key in self.active_map:

            return self.active_map[key]

        

        need = self.wait_map[key]['file_size']

        onnx = self.wait_map[key]['onnx']



        # check current memory use

        used_size, biggest_k = self.used()

        while biggest_k is not None and self.max_size - used_size < need:

            # if exceeded once, delete until `max(half_max, file_size)` left

            need = max(need, self.max_size / 2)

            if len(self.active_map) == 0:

                break



            del self.active_map[biggest_k]

            used_size, biggest_k = self.used()

        

        self.active_map[key] = OrtWrapper(onnx)

        return self.active_map[key]

