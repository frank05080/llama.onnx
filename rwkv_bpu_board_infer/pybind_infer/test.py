from bpu_infer_lib import infer

ptr1, ptr2 = infer("/root/rwkv/models/rwkv_mixing_0.bin", "/root/rwkv/inputs/inputs_0.bin", "/root/rwkv/inputs/inputs_1.bin")

array1 = [ptr1[i] for i in range(5)]
array2 = [ptr2[i] for i in range(5)]

print(array1)
print(type(array1))
print(array2)