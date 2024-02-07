import huffman
import math


def huffman_v2(gateset_basic: dict):
    # cbits = math.ceil(math.log2(len(gateset_basic)))
    # # print(cbits)
    # numbits = 0
    # for instruc in gateset_basic:
    #     numbits += gateset_basic[instruc] * cbits

    frequencies = sorted(gateset_basic.items(), key=lambda x: x[1], reverse=True)
    huff = huffman.huffman(gateset_basic, frequencies)
    sumbits = 0

    # print(' %-30r |%43s' % ('Gate', 'Huffman Code'))
    # print('-----------------------------------------------------------------------------')
    for (char, frequency) in frequencies:
        # print(' %-30r |%43s' % (char, huff[char]))
        sumbits += frequency * len(huff[char])

    # print("Bits needed with standard 6-bit encoding of composite instructions of the basic approximation set: ",
    #       numbits)
    # print("Bits needed after Huffman coding of composite instructions of the basic approximation set: ", sumbits)
    # print("Compression Ratio (length of encoded bit-string/length of bit-string): ", sumbits / numbits)

    # return sumbits / numbits, huff
    return sumbits, huff
