import numpy as np
# 加载地平线依赖库
from horizon_tc_ui import HB_ONNXRuntime
# 准备模型运行的输入, 生成形状为 (1,) 的 int32 数据
input_data = np.random.randint(low=0, high=100, size=(1,), dtype=np.int32)
# 加载模型文件
sess = HB_ONNXRuntime(model_file = "optimized_embed.onnx")
# 获取模型输入&输出节点信息
input_names = sess.input_names
output_names = sess.output_names
# 准备输入数据，这里我们假设此模型只有1个输入
input_info = {input_names[0]: input_data}
# 开始模型推理，推理的返回值是一个list，依次与output_names指定名称一一对应
output = sess.run(output_names, input_info)
print(output)