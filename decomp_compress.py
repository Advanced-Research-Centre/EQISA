# UTILITIES
import pathlib
import numpy as np
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# QISKIT DEPENDENCIES
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary, Statevector, Choi, process_fidelity, Operator, entropy, DensityMatrix
import qiskit.qasm2 as qasm

# DEPENDENCIES FROM FILE
from v0_binary import binary_encoding
from v1_huff_gateset import huffman_v1
from v2_huff_skbasis import huffman_v2
from v3_huff_select_skbasis import huffman_v3
from custom_basis import prep_gateset, _1q_gates, _1q_inverses
from custom_sk_decomp import SolovayKitaevDecomposition
from helper import Helper


class SKDecompose:
    """
    Class for handling the decomposition and compression routine for 1q unitaries.
    For a dataset of 1q Harr-random unitaries, perform Solovay-Kitaev Decomposition
    and Huffman encoding of instruction stream
    """
    def encoding(self, dict_gateset: dict, dict_basis: dict, depth: int):
        """
        args
        :param dict_gateset: dictionary with gates and their frequencies
        :param dict_basis: dictionary with SK basis instructions and their frequencies
        :param depth: depth of SK basis
        :return: compression factors for Huffman v1, v2 and v3 respectively
        """
        # v0: Binary Encoding of native gates
        # sumbits0 = 2 * sum(list(dict_gateset.values()))
        sumbits0 = binary_encoding(dict_gateset)

        # v1: Huffman Encoding of the gate-set
        sumbits1, huff1 = huffman_v1(dict_gateset)
        c_factor1 = sumbits1 / sumbits0
        # c_ratio1 = sumbits0 / sumbits1

        # v2: Huffman Encoding of the S-K Basis
        sumbits2, huff2 = huffman_v2(dict_basis)
        c_factor2 = sumbits2 / sumbits0
        # c_ratio2 = sumbits0 / sumbits2

        # v3: Huffman Encoding of selected instructions form the S-K Basis
        filename = '../selections/selection_depth' + str(depth) + '_1q.txt'
        # filename = '../selections/selection_depth4_1q.txt'
        sumbits3, huff3 = huffman_v3(dict_basis, filename)
        c_factor3 = sumbits3 / sumbits0
        # c_ratio3 = sumbits0 / sumbits3
        return c_factor1, c_factor2, c_factor3

    def decompose(self,
                  unitary: np.ndarray,
                  gateset: list,
                  recur: int,
                  depth: int,
                  save_qasm: bool = False,
                  save_streams: bool = False,
                  param: int = 0):
        """
        :param unitary: unitary matrix to decompose
        :param gateset: list of basis gates. Example: ['h', 't']. Doesn't need to be closed under inversion
        :param recur: Degree of recursion for Solovay-Kitaev decomposition (SKD)
        :param depth: depth of SK basis for SKD
        :param save_qasm: Boolean; True for saving the qasm code for decomposed circuit. default=False
        :param save_streams: Boolean; True for saving the stream of SK basis instructions. default=False
        :param param: integer parameter for saving the qasm and stream with index number of the sample in the dataset
        :return: compression factors for Huffman v1, v2 and v3, process fidelity and circuit_depth for decomposition
        """
        basis = prep_gateset(gateset, depth)
        skt = SolovayKitaevDecomposition(d=depth, basic_approximations=basis)
        skt.__init__(d=depth, basic_approximations=basis)
        circ, dict_gateset, find_basics = skt.run(gate_matrix=unitary, recursion_degree=recur)
        choi0 = Choi(UnitaryGate(unitary))
        choi_circ = Choi(circ)
        fidelity = process_fidelity(choi_circ, choi0)
        circuit_depth = circ.depth()
        elements = [instruc.labels for instruc in basis]
        dict_sk_basis = {str(element): 0 for element in elements}
        help = Helper()
        help.__int__(dict_sk_basis, find_basics, recur)
        sk_basis = help.basic_traverse()
        instruction_stream = help.call_nested()
        del sk_basis['[]']
        direc = '../Data/decomp_circ_db/1q/d' + str(depth) + '_r' + str(recur) + '/'
        filename = '1q_' + f'{param:03}' + '.txt'
        path = pathlib.Path(direc + filename)

        if save_qasm:
            qasm.dump(circ, path)

        if save_streams:
            path2 = '../Data/decomp_circ_db/1q/d' + str(depth) + '_r' + str(recur) + '_streams/'
            file = '1q_' + f'{param:03}' + '.txt'
            f = open(path2 + file, 'w')
            for inst in instruction_stream:
                f.write(inst + ' ' + str(0) + '\n')

        cf1, cf2, cf3 = self.encoding(dict_gateset, sk_basis, depth)

        return cf1, cf2, cf3, fidelity, circuit_depth
        # return fidelity, circuit_depth

    def caller(self, gateset: list, depth: int, recur: int, write_to_records: bool=False) -> None:
        """
        :param gateset: list of gates in the native set
        :param depth: depth of Solovay-Kitaev basis
        :param recur: degree of recursion for SKD
        :param write_to_records: boolean for writing to csv file in records. default=False
        :return: None
        """
        numtrials = 200
        direc = '/Data/unitary_db/1q/'
        u_list = np.empty(numtrials, dtype=np.ndarray)
        for filename in os.listdir(direc):
            ind = int(filename[7:10])
            u_list[ind] = np.load(direc + '/' + filename)

        fidelities = np.zeros(numtrials)
        cds = np.zeros(numtrials)
        huff1_cr = np.zeros(numtrials)
        huff2_cr = np.zeros(numtrials)
        huff3_cr = np.zeros(numtrials)

        if write_to_records:
            records_direc = '/records/1q/'
            filename = '1q_huff_records.csv'
            file = open(records_direc + filename, 'a', newline='')
            writer = csv.writer(file)
            writer.writerow(['depth', 'recur', 'mean fidelity', 'std fidelity', 'mean circ_depth', ' std circ_depth',
                             'mean huff1 cr', 'std huff1 cr', 'mean huff2 cr', 'std huff2 cr', 'mean huff3 cr',
                             'std huff3 cr'])

        for i in range(numtrials):
            c_factor1, c_factor2, c_factor3, fidelity, cd = self.decompose(unitary=u_list[i],
                                                                          gateset=gateset,
                                                                          recur=recur,
                                                                          depth=depth,
                                                                          save_qasm=True,
                                                                          save_streams=True,
                                                                          param=i)
            fidelities[i] = fidelity
            cds[i] = cd
            huff1_cr[i] = c_factor1
            huff2_cr[i] = c_factor2
            huff3_cr[i] = c_factor3
        #
        print("For depth = ", depth, "; recur = ", recur)
        print("Fidelity = ", np.average(fidelities), " ± ", np.std(fidelities))
        print("Circuit Depth = ", np.average(cds), " ± ", np.std(cds))
        print("Mean Huff1 CR = ", np.average(huff1_cr), " ± ", np.std(huff1_cr))
        print("Mean Huff2 CR = ", np.average(huff2_cr), " ± ", np.std(huff2_cr))
        print("Mean Huff3 CR = ", np.average(huff3_cr), " ± ", np.std(huff3_cr))
        #
        if write_to_records:
            writer.writerow([depth, recur, np.average(fidelities), np.std(fidelities), np.average(cds), np.std(cds),
                         np.average(huff1_cr), np.std(huff1_cr), np.average(huff2_cr), np.std(huff2_cr),
                         np.average(huff3_cr), np.std(huff3_cr)])



skd = SKDecompose()
depth = 5
# recur = 4
# depth_arr = [1, 2, 3]
recur_arr = [6]
gateset = ['h', 't']  # standard gateset adopted throughout the project
# gateset = ['p1a', 'p1b'] # special optimal gates from YAQQ
for recur in recur_arr:
    skd.caller(gateset, depth, recur)

