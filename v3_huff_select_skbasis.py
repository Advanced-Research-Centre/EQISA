import huffman
import math
import numpy as np
from ast import literal_eval


def modif_dict_v3(gateset_basic, filename):
    basic = gateset_basic.copy()
    basic_approx_sym = list(basic.keys())
    basic_approx_freq = list(basic.values())

    with open(filename, 'r') as file:
        info = file.readlines()
        selected_sym = [info[index].strip() for index in range(len(info))]
        # print(selected_sym)
    for i in range(len(basic_approx_sym)):
        if (basic_approx_sym[i] not in selected_sym and basic_approx_sym[i] != "['h']"
                and basic_approx_sym[i] != "['t']" and basic_approx_sym[i] != "['tdg']"):
            if basic_approx_freq[i] != 0:
                del basic[basic_approx_sym[i]]
                string_arr = literal_eval(basic_approx_sym[i])
                for gates in string_arr:
                    if gates == 'h':
                        basic["['h']"] += 1 * basic_approx_freq[i]
                    elif gates == 't':
                        basic["['t']"] += 1 * basic_approx_freq[i]
                    elif gates == 'tdg':
                        basic["['tdg']"] += 1 * basic_approx_freq[i]
            else:
                del basic[basic_approx_sym[i]]

    return basic


def huffman_v3(gateset_basic: dict, filename: str):
    # print("Original dict given = ", gateset_basic)
    # cbits = math.ceil(math.log2(len(gateset_basic)))
    # print(cbits)
    # numbits = 0
    # for instruc in gateset_basic:
    #     numbits += gateset_basic[instruc] * cbits

    modif_basic = modif_dict_v3(gateset_basic, filename)
    # print("After modification, dict = ", modif_basic)
    frequencies = sorted(modif_basic.items(), key=lambda x: x[1], reverse=True)
    huff = huffman.huffman(modif_basic, frequencies)
    sumbits = 0
    # csum = 0
    # print(' %-30r |%43s' % ('Gate', 'Huffman Code'))
    # print('-----------------------------------------------------------------------------')
    for (char, frequency) in frequencies:
        # print(' %-30r |%43s' % (char, huff[char]))
        sumbits += frequency * len(huff[char])
    #     csum += frequency
    # print("Huffman v3 no. of instructions = ", csum)
    # print("Bits needed with standard ", cbits, "-bit encoding of composite instructions of the basic approximation set: ",
    #       numbits)
    # print("Bits needed after Huffman coding of composite instructions of the basic approximation set: ", sumbits)
    # print("Compression Ratio (length of encoded bit-string/length of bit-string): ", sumbits / numbits)

    return sumbits, huff