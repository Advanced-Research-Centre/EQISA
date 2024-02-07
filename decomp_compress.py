import numpy as np
from qiskit.extensions import UnitaryGate
import os
import csv
import warnings
warnings.filterwarnings("ignore")
from qiskit.quantum_info import random_unitary, Statevector, Choi, process_fidelity, Operator, entropy, DensityMatrix
from v0_binary import binary_encoding
from v1_huff_gateset import huffman_v1
from v2_huff_skbasis import huffman_v2
from v3_huff_select_skbasis import huffman_v3
from custom_basis import prep_gateset, _1q_gates, _1q_inverses
from custom_sk_decomp import SolovayKitaevDecomposition
from helper import Helper


def decompose(unitary, gateset, recur, depth):
    basis_approximation = prep_gateset(gateset, depth)
    skt = SolovayKitaevDecomposition(d=depth, basic_approximations=basis_approximation)
    skt.__init__(d=depth, basic_approximations=basis_approximation)
    circ, dict_gateset, find_basics = skt.run(gate_matrix=unitary, recursion_degree=recur)
    choi0 = Choi(UnitaryGate(unitary))
    choi_circ = Choi(circ)
    fidelity = process_fidelity(choi_circ, choi0)
    circuit_depth = circ.depth()
    basis_elements = [basis_approximation[i].labels for i in range(len(basis_approximation))]
    help = Helper()
    dict_sk_basis = {str(basis_elements[i]): 0 for i in range(len(basis_elements))}
    help.__int__(dict_sk_basis, find_basics, recur)
    sk_basis = help.basic_traverse()
    instruction_stream = help.call_nested()
    del sk_basis['[]']
    # v0: Binary Encoding of native gates
    sumbits0 = binary_encoding(dict_gateset)
    # v1: Huffman Encoding of the gate-set
    sumbits1, huff1 = huffman_v1(dict_gateset)
    c_factor1 = sumbits1 / sumbits0
    c_ratio1 = sumbits0 / sumbits1
    # v2: Huffman Encoding of the S-K Basis
    sumbits2, huff2 = huffman_v2(sk_basis)
    c_factor2 = sumbits2 / sumbits0
    c_ratio2 = sumbits0 / sumbits2
    # v3: Huffman Encoding of selected instructions form the S-K Basis
    sumbits3, huff3 = huffman_v3(sk_basis)
    c_factor3 = sumbits3 / sumbits0
    c_ratio3 = sumbits0 / sumbits3
    return dict_gateset, sk_basis, c_factor1, c_factor2, c_factor3

numtrials = 200
direc = 'Data/Unitaries_200/Unitaries'
u_list = np.empty(numtrials, dtype=np.ndarray)
gateset = ['h', 't']
depth = 4
recur = 3
for filename in os.listdir(direc):
    ind = int(filename[7:10])
    u_list[ind] = np.load(direc+'/'+filename)

with open('SK-Basis.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(numtrials):
        dict_gateset, sk_basis, cf1, cf2, cf3 = decompose(u_list[i], gateset, depth, recur)
        if i == 0:
            writer.writerow(list(sk_basis.keys()))
        writer.writerow(list(sk_basis.values()))




