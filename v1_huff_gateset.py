import huffman
import math

def huffman_v0(gateset_list: dict):
    # cbits = math.ceil(math.log2(len(gateset_list)))
    # # print(cbits)
    # numbits = 0
    # for gate in gateset_list:
    #     numbits += gateset_list[gate] * cbits

    frequencies = sorted(gateset_list.items(), key=lambda x: x[1], reverse=True)
    huff = huffman.huffman(gateset_list, frequencies)
    sumbits = 0

    # print(' %-30r |%43s' % ('Gate', 'Huffman Code'))
    # print('-----------------------------------------------------------------------------')
    for (char, frequency) in frequencies:
        # print(' %-30r |%43s' % (char, huff[char]))
        # if math.isnan(sumbits_gateset / numbits) is False:
        sumbits += frequency * len(huff[char])

    # print("Bits needed with standard 6-bit encoding of composite instructions of the basic approximation set: ",
    #       numbits)
    # print("Bits needed after Huffman coding of composite instructions of the basic approximation set: ", sumbits)
    # print("Compression Ratio (length of encoded bit-string/length of bit-string): ", sumbits / numbits)
    # if numbits != 0:
    return sumbits, huff
