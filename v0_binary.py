# v0 Binary encoding of decomposed circuit

import math

def binary_encoding(gateset_list: dict):
    # Number of bits used for encoding = cbits
    # if len(gateset_list) == 1:
    #     cbits = 2
    cbits = math.ceil(math.log2(len(gateset_list)))
    # print("Number of bits per gate for binary encoding: ", cbits)
    size = 0
    for gate in gateset_list:
        size += gateset_list[gate] * cbits

    return size
