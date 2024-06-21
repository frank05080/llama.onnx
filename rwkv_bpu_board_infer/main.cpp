#include <iostream>
#include <string>
#include <vector>
#include <memory>             // include this to use std::shared_ptr
#include <chrono>             // include this to use std::chrono::system_clock::now()
#include <mutex>              // include this to use std::mutex
#include <condition_variable> // include this to use std::condition_variable

#include "dnn/hb_dnn.h"

bool check_ret(int32_t ret)
{
    if (ret != 0)
    {
        std::cerr << "Failed to get model name list, error code: " << ret << std::endl;
        return false;
    }
    return true;
}

void checkTensorProperties(hbDNNTensor tensor){
    std::cout << "### start prop check: " << std::endl;
    std::cout << "tensor has type: " << tensor.properties.tensorType << std::endl;
    std::cout << "prop.tensorLayout: " << tensor.properties.tensorLayout << std::endl;
    std::cout << "validShape numDimensions: " << tensor.properties.validShape.numDimensions << std::endl;
    for (int j = 0; j < tensor.properties.validShape.numDimensions; ++j) {
        std::cout << "dimensionSize[" << j << "]: " << tensor.properties.validShape.dimensionSize[j] << std::endl;
    }
    std::cout << "alignedShape numDimensions: " << tensor.properties.alignedShape.numDimensions << std::endl;
    for (int j = 0; j < tensor.properties.alignedShape.numDimensions; ++j) {
        std::cout << "dimensionSize[" << j << "]: " << tensor.properties.alignedShape.dimensionSize[j] << std::endl;
    }
}

int prepare_tensor(hbDNNTensor *input_tensors, hbDNNTensor *output_tensor, hbDNNHandle_t dnn_handle, int input_count, int output_count)
{
    hbDNNTensor *inputs = input_tensors;
    // auto prop = input_tensors[0].properties; // get default value of properties.
    // std::cout << prop.tensorLayout << std::endl;
    auto ret = -1;
    for(int i=0; i<input_count; i++) {
        ret = hbDNNGetInputTensorProperties(&inputs[i].properties, dnn_handle, i); // get input_tensors[0]'s ptr
        check_ret(ret);
        checkTensorProperties(inputs[i]);
        int input_memSize = inputs[i].properties.alignedByteSize;
        ret = hbSysAllocCachedMem(&inputs[i].sysMem[0], input_memSize);
        check_ret(ret);
    }
}

int main(int argc, char **argv)
{
    // 1. 加载模型
    hbPackedDNNHandle_t packed_dnn_handle;
    hbDNNHandle_t dnn_handle;
    char const *modelFileNames[] = {
        "/root/rwkv/rwkv_mixing_0.bin",
    };
    const char **model_name_list;
    int model_count = 0;
    auto ret = hbDNNInitializeFromFiles(&packed_dnn_handle, modelFileNames, 1);
    check_ret(ret);
    ret = hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    check_ret(ret);
    std::cout << "Model count: " << model_count << std::endl;
    for (int i = 0; i < model_count; ++i)
    {
        std::cout << "Model " << i << ": " << model_name_list[i] << std::endl;
    }
    ret = hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);
    check_ret(ret);

    // // 2. 获取input/output信息
    std::vector<hbDNNTensor> input_tensors;
    std::vector<hbDNNTensor> output_tensors;

    int input_count = 0;
    int output_count = 0;
    ret = hbDNNGetInputCount(&input_count, dnn_handle);
    check_ret(ret);
    std::cout << "input_count is: " << input_count << std::endl;
    ret = hbDNNGetOutputCount(&output_count, dnn_handle);
    check_ret(ret);
    std::cout << "output_count is: " << output_count << std::endl;
    input_tensors.resize(input_count);
    output_tensors.resize(output_count);

    prepare_tensor(input_tensors.data(), output_tensors.data(), dnn_handle, input_count, output_count);
    
    return 0;
}
