#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>             // include this to use std::shared_ptr
#include <chrono>             // include this to use std::chrono::system_clock::now()
#include <mutex>              // include this to use std::mutex
#include <condition_variable> // include this to use std::condition_variable
#include <cstring>            // for memcpy
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic conversion between C++ STL and Python

#include "dnn/hb_dnn.h"

namespace py = pybind11;
bool DEBUG = false;
// std::vector<std::string> model_files_;
std::vector<hbPackedDNNHandle_t> packed_dnn_handles_;
std::vector<hbDNNHandle_t> dnn_handles_;

int INPUT_MODE = 0;
int OUTPUT_MODE = 1;

bool check_ret(int32_t ret)
{
    if (ret != 0)
    {
        std::cerr << "Failed to get model name list, error code: " << ret << std::endl;
        return false;
    }
    return true;
}

void checkTensorProperties(hbDNNTensor tensor)
{
    std::cout << "### start prop check: " << std::endl;
    std::cout << "tensor has type: " << tensor.properties.tensorType << std::endl;
    std::cout << "prop.tensorLayout: " << tensor.properties.tensorLayout << std::endl;
    std::cout << "validShape numDimensions: " << tensor.properties.validShape.numDimensions << std::endl;
    for (int j = 0; j < tensor.properties.validShape.numDimensions; ++j)
    {
        std::cout << "dimensionSize[" << j << "]: " << tensor.properties.validShape.dimensionSize[j] << std::endl;
    }
    std::cout << "alignedShape numDimensions: " << tensor.properties.alignedShape.numDimensions << std::endl;
    for (int j = 0; j < tensor.properties.alignedShape.numDimensions; ++j)
    {
        std::cout << "dimensionSize[" << j << "]: " << tensor.properties.alignedShape.dimensionSize[j] << std::endl;
    }
}

void applyTensorMem(hbDNNTensor *tensors, int count, hbDNNHandle_t dnn_handle, int mode)
{
    auto ret = -1;
    for (int i = 0; i < count; i++)
    {
        if (mode == INPUT_MODE)
        {
            ret = hbDNNGetInputTensorProperties(&tensors[i].properties, dnn_handle, i); // get input_tensors[0]'s ptr
        }
        else
        {
            ret = hbDNNGetOutputTensorProperties(&tensors[i].properties, dnn_handle, i);
        }
        check_ret(ret);
        if(DEBUG){
            checkTensorProperties(tensors[i]);
        }

        int memSize = tensors[i].properties.alignedByteSize;
        if(DEBUG){
            std::cout << "memSize: " << memSize << std::endl;
        }
        ret = hbSysAllocCachedMem(&tensors[i].sysMem[0], memSize);
        check_ret(ret);
    }
}

std::vector<float> read_binary_file(const std::string &filepath, size_t num_elements)
{
    std::vector<float> data(num_elements);
    std::ifstream file(filepath, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    if(DEBUG){
        std::cout << "sizeof(float) is: " << sizeof(float) << std::endl;
    }
    file.read(reinterpret_cast<char *>(data.data()), num_elements * sizeof(float));
    return data;
}

int prepare_tensor(hbDNNTensor *input_tensors, hbDNNTensor *output_tensors, hbDNNHandle_t dnn_handle, int input_count, int output_count)
{
    hbDNNTensor *inputs = input_tensors;
    // auto prop = input_tensors[0].properties; // get default value of properties.
    // std::cout << prop.tensorLayout << std::endl;
    applyTensorMem(inputs, input_count, dnn_handle, INPUT_MODE);
    hbDNNTensor *outputs = output_tensors;
    applyTensorMem(outputs, output_count, dnn_handle, OUTPUT_MODE);
}

std::pair<std::vector<float>, std::vector<float>> mixing_infer(std::string model_bin_path, std::string input0_path, std::string input1_path, int ind)
{
    std::cout << "ind: " << ind << std::endl;
    // hbPackedDNNHandle_t packed_dnn_handle;
    // hbDNNHandle_t dnn_handle;
    // char const *modelFileNames[] = {
    //     model_bin_path.c_str()
    // };
    // const char **model_name_list;
    // int model_count = 0;
    // std::cout << "try to init" << std::endl;
    // auto ret = hbDNNInitializeFromFiles(&packed_dnn_handle, modelFileNames, 1);
    // std::cout << "finish init" << std::endl;
    // check_ret(ret);
    // std::cout << "try to hbDNNGetModelNameList" << std::endl;
    // ret = hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    // check_ret(ret);
    // if(DEBUG){
    //     std::cout << "Model count: " << model_count << std::endl;
    //     for (int i = 0; i < model_count; ++i)
    //     {
    //         std::cout << "Model " << i << ": " << model_name_list[i] << std::endl;
    //     }
    // }
    // ret = hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);
    // check_ret(ret);
    std::cout << "dnn_handle is nullptr: " << (dnn_handles_[ind]== nullptr) << std::endl;

    std::vector<hbDNNTensor> input_tensors;
    std::vector<hbDNNTensor> output_tensors;

    int input_count = 0;
    int output_count = 0;
    auto ret = hbDNNGetInputCount(&input_count, dnn_handles_[ind]);
    check_ret(ret);
    if(DEBUG){
        std::cout << "input_count is: " << input_count << std::endl;
    }
    ret = hbDNNGetOutputCount(&output_count, dnn_handles_[ind]);
    check_ret(ret);
    if(DEBUG){
        std::cout << "output_count is: " << output_count << std::endl;
    }
    input_tensors.resize(input_count);
    output_tensors.resize(output_count);

    prepare_tensor(input_tensors.data(), output_tensors.data(), dnn_handles_[ind], input_count, output_count);

    // 3. 读取bin
    std::string filepath = input0_path;
    size_t num_elements = 1024;
    std::vector<float> input0_data;
    try
    {
        input0_data = read_binary_file(filepath, num_elements);
        // std::cout << "Read " << input0_data.size() << " elements from " << filepath << std::endl;
        // Print the data (optional)
        // for (size_t i = 0; i < data.size(); ++i) {
        //     std::cout << data[i] << " ";
        //     if ((i + 1) % 10 == 0) { // Print 10 elements per line
        //         std::cout << std::endl;
        //     }
        // }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }

    // 4. 将.bin写入tensor
    auto vir_addr = input_tensors[0].sysMem[0].virAddr;
    std::memcpy(vir_addr, input0_data.data(), input0_data.size() * sizeof(float)); // pass the pointer to the data contained in the vector
    // Check the size in vir_addr by printing out the values (Optional)

    float *vir_addr_float = reinterpret_cast<float *>(input_tensors[0].sysMem[0].virAddr); // vir_addr is void*
    if (DEBUG){
        std::cout << "Data in vir_addr:" << std::endl;
        for (size_t i = 0; i < 100; ++i)
        { // input0_data.size()
            std::cout << vir_addr_float[i] << " ";
            if ((i + 1) % 10 == 0)
            { // Print 10 elements per line
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }

    filepath = input1_path;
    std::vector<float> input1_data;
    int rows = 5;
    int cols = 1024;
    try
    {
        input1_data = read_binary_file(filepath, num_elements * 5);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }

    // 4. 将.bin写入tensor
    vir_addr = input_tensors[1].sysMem[0].virAddr;
    std::memcpy(vir_addr, input1_data.data(), input1_data.size() * sizeof(float));

    if (DEBUG){
        vir_addr_float = reinterpret_cast<float *>(input_tensors[1].sysMem[0].virAddr); // vir_addr is void*
        std::cout << "Data in vir_addr:" << std::endl;
        for (size_t i = 0; i < 100; ++i)
        { // input0_data.size()
            std::cout << vir_addr_float[i] << " ";
            if ((i + 1) % 10 == 0)
            { // Print 10 elements per line
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }

    // 5. 执行推理
    std::cout << "about to infer: " << ind << std::endl;
    hbDNNTaskHandle_t task_handle = nullptr;
    hbDNNTensor *output = output_tensors.data();

    for (int i = 0; i < input_count; i++)
    {
        hbSysFlushMem(&input_tensors[i].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    }

    hbDNNInferCtrlParam infer_ctl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctl_param);

    hbDNNInfer(&task_handle, &output, input_tensors.data(), dnn_handles_[ind], &infer_ctl_param);
    hbDNNWaitTaskDone(task_handle, 0);

    // 6. 后处理
    for (int i = 0; i < output_count; i++)
    {
        hbSysFlushMem(&output_tensors[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    }

    float *out_vir_addr_float = reinterpret_cast<float *>(output_tensors[0].sysMem[0].virAddr); // vir_addr is void*
    if(DEBUG){
        std::cout << "Data in vir_addr:" << std::endl;
        for (size_t i = 0; i < 100; ++i)
        {
            std::cout << out_vir_addr_float[i] << " ";
            if ((i + 1) % 10 == 0)
            { // Print 10 elements per line
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }
    
    float *out_vir_addr_float2 = reinterpret_cast<float *>(output_tensors[1].sysMem[0].virAddr); // vir_addr is void*
    if (DEBUG){
        std::cout << "Data in vir_addr:" << std::endl;
        for (size_t i = 0; i < 100; ++i)
        {
            std::cout << out_vir_addr_float2[i] << " ";
            if ((i + 1) % 10 == 0)
            { // Print 10 elements per line
                std::cout << std::endl;
            }
        }
    }

    std::vector<float> vec1(out_vir_addr_float, out_vir_addr_float + 1024 * sizeof(float));
    std::vector<float> vec2(out_vir_addr_float2, out_vir_addr_float2 + 5 * 1024 * sizeof(float));

    // hbDNNReleaseTask(task_handle);
    // for(int i=0; i<input_count; i++){
    //     hbSysFreeMem(&(input_tensors[i].sysMem[0]));
    // }
    // for(int i=0; i<input_count; i++){
    //     hbSysFreeMem(&(output_tensors[i].sysMem[0]));
    // }
    // hbDNNReleaseTask(packed_dnn_handle[i]);
    
    return std::make_pair(vec1, vec2);
}

std::vector<float> head_infer(std::string model_bin_path, std::string input0_path, int ind)
{
    // hbPackedDNNHandle_t packed_dnn_handle;
    // hbDNNHandle_t dnn_handle;
    // char const *modelFileNames[] = {
    //     model_bin_path.c_str()
    // };
    // const char **model_name_list;
    // int model_count = 0;
    // auto ret = hbDNNInitializeFromFiles(&packed_dnn_handle, modelFileNames, 1);
    // check_ret(ret);
    // ret = hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    // check_ret(ret);
    // if(DEBUG){
    //     std::cout << "Model count: " << model_count << std::endl;
    //     for (int i = 0; i < model_count; ++i)
    //     {
    //         std::cout << "Model " << i << ": " << model_name_list[i] << std::endl;
    //     }
    // }
    // ret = hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);
    // check_ret(ret);

    std::vector<hbDNNTensor> input_tensors;
    std::vector<hbDNNTensor> output_tensors;

    int input_count = 0;
    int output_count = 0;
    auto ret = hbDNNGetInputCount(&input_count, dnn_handles_[ind]);
    check_ret(ret);
    if (DEBUG){
        std::cout << "input_count is: " << input_count << std::endl;
    }
    ret = hbDNNGetOutputCount(&output_count, dnn_handles_[ind]);
    check_ret(ret);
    if (DEBUG){
        std::cout << "output_count is: " << output_count << std::endl;
    }
    input_tensors.resize(input_count);
    output_tensors.resize(output_count);

    prepare_tensor(input_tensors.data(), output_tensors.data(), dnn_handles_[ind], input_count, output_count);

    // 3. 读取bin
    std::string filepath = input0_path;
    size_t num_elements = 1024;
    std::vector<float> input0_data;
    try
    {
        input0_data = read_binary_file(filepath, num_elements);
        // std::cout << "Read " << input0_data.size() << " elements from " << filepath << std::endl;
        // Print the data (optional)
        // for (size_t i = 0; i < data.size(); ++i) {
        //     std::cout << data[i] << " ";
        //     if ((i + 1) % 10 == 0) { // Print 10 elements per line
        //         std::cout << std::endl;
        //     }
        // }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }

    // 4. 将.bin写入tensor
    auto vir_addr = input_tensors[0].sysMem[0].virAddr;
    std::memcpy(vir_addr, input0_data.data(), input0_data.size() * sizeof(float)); // pass the pointer to the data contained in the vector
    // Check the size in vir_addr by printing out the values (Optional)
    float *vir_addr_float = reinterpret_cast<float *>(input_tensors[0].sysMem[0].virAddr); // vir_addr is void*

    if (DEBUG){
        std::cout << "Data in vir_addr:" << std::endl;
        for (size_t i = 0; i < 100; ++i)
        { // input0_data.size()
            std::cout << vir_addr_float[i] << " ";
            if ((i + 1) % 10 == 0)
            { // Print 10 elements per line
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }
    
    // 5. 执行推理
    hbDNNTaskHandle_t task_handle = nullptr;
    hbDNNTensor *output = output_tensors.data();

    for (int i = 0; i < input_count; i++)
    {
        hbSysFlushMem(&input_tensors[i].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    }

    hbDNNInferCtrlParam infer_ctl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctl_param);

    hbDNNInfer(&task_handle, &output, input_tensors.data(), dnn_handles_[ind], &infer_ctl_param);
    hbDNNWaitTaskDone(task_handle, 0);

    // 6. 后处理
    for (int i = 0; i < output_count; i++)
    {
        hbSysFlushMem(&output_tensors[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    }

    float *out_vir_addr_float = reinterpret_cast<float *>(output_tensors[0].sysMem[0].virAddr); // vir_addr is void*

    if (DEBUG){
        std::cout << "Data in vir_addr:" << std::endl;
        for (size_t i = 0; i < 100; ++i)
        {
            std::cout << out_vir_addr_float[i] << " ";
            if ((i + 1) % 10 == 0)
            { // Print 10 elements per line
                std::cout << std::endl;
            }
        }
    }

    std::cout << "ready to fin_ret" << std::endl;
    std::vector<float> fin_ret(out_vir_addr_float, out_vir_addr_float + 50277 * sizeof(float));

    // hbDNNReleaseTask(task_handle);
    // for(int i=0; i<input_count; i++){
    //     hbSysFreeMem(&(input_tensors[i].sysMem[0]));
    // }
    // for(int i=0; i<input_count; i++){
    //     hbSysFreeMem(&(output_tensors[i].sysMem[0]));
    // }
    // hbDNNReleaseTask(packed_dnn_handle);

    return fin_ret;
}

// Function to initialize DNN handles and return a list of them
void initialize_dnn_handles(std::vector<std::string> model_files)
{
    for (const auto &model_file : model_files)
    {
        hbPackedDNNHandle_t packed_dnn_handle;
        hbDNNHandle_t dnn_handle;
        const char *modelFileNames[] = {model_file.c_str()};
        const char **model_name_list;
        int model_count = 0;

        int ret = hbDNNInitializeFromFiles(&packed_dnn_handle, modelFileNames, 1);
        if (!check_ret(ret)) continue;

        ret = hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
        if (!check_ret(ret)) continue;

        std::cout << "Model count: " << model_count << std::endl;
        for (int i = 0; i < model_count; ++i)
        {
            std::cout << "Model " << i << ": " << model_name_list[i] << std::endl;
        }

        ret = hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);
        if (!check_ret(ret)) continue;

        dnn_handles_.push_back(dnn_handle);
        packed_dnn_handles_.push_back(packed_dnn_handle);
    }
    std::cout << "finished init dnn handles" << std::endl;
}

PYBIND11_MODULE(bpu_infer_lib, m)
{
    m.doc() = "bpu_infer_lib plugin"; // optional module docstring

    m.def("mixing_infer", &mixing_infer, "A function that adds two numbers");
    m.def("head_infer", &head_infer, "A function that adds two numbers");
    m.def("initialize_dnn_handles", &initialize_dnn_handles, "A function that adds two numbers");
}