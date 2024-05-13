# UTILITIES

import numpy as np
import pathlib
import warnings
warnings.filterwarnings("ignore")

# QISKIT DEPENDENCIES
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.quantumcircuit import QuantumCircuit
import qiskit.qasm2 as qasm
from qiskit.compiler.transpiler import transpile
from qiskit.quantum_info import random_unitary

# DEPENDENCIES FROM FILE
from custom_sk_decomp import SolovayKitaevDecomposition
from qiskit_qsd import qs_decomposition
from custom_basis import prep_gateset, identify_gates, define_gateset
from helper import Helper


class QuantumShannon_SKT():
    """
    Class for performing the decomposition of multi-qubit unitaries.
    Decomposition method: QSD + SKD
    QSD: decomposes unitary into CX gates and single qubit unitary rotations
    SKD: decomposes single qubit unitary rotations into native gates ['h', 't', tdg']
    """
    def __int__(self, dim):
        self.dim = dim

    def decompose(self, unitary: np.ndarray,
                  d: int,
                  n: int,
                  gateset: list[str],
                  save_qasm: bool = False,
                  save_streams: bool = False,
                  param: str = ""):
        qstream = []
        istream = []
        circ = qs_decomposition(unitary, opt_a1=False, opt_a2=True)
        decomp_circ = QuantumCircuit(self.dim, global_phase=circ.global_phase)
        _1q_gates, _2q_gates = identify_gates(gateset)
        basis = prep_gateset(_1q_gates, d)
        elements = [instruc.labels for instruc in basis]
        tot_sk_basis = {str(element): 0 for element in elements}
        del tot_sk_basis['[]']
        tot_sk_basis["['cx']"] = 0
        tot_gateset = {'h': 0,
                       't': 0,
                       'tdg': 0,
                       'cx': 0}
        if self.dim > 2:
            circ = transpile(circ, basis_gates=['u3', 'cx'])  # Necessary to unroll custom gates in decomposed circuits
        # print(circ)
        for gate in circ:
            if gate.operation.num_qubits == 1:
                qubit_id = gate.qubits[0]._index
                mat = np.array(UnitaryGate(gate.operation).to_matrix(), dtype=complex)
                subcirc, instruc_stream, dict_basic = self.skt_decomp(mat, d, n, basis)
                istream.extend(instruc_stream)
                q = [qubit_id] * len(instruc_stream)
                qstream.extend(q)
                for instruc in dict_basic.keys():
                    tot_sk_basis[instruc] += dict_basic[instruc]
                for subgate in subcirc:
                    decomp_circ.append(subgate.operation, [qubit_id])
                    tot_gateset[subgate.operation.name] += 1
            else:
                qubit_id = [gate.qubits[0]._index, gate.qubits[1]._index]
                qstream.append(qubit_id[0])
                qstream.append(qubit_id[1])
                istream.append("['cx']")
                tot_gateset['cx'] += 1
                tot_sk_basis["['cx']"] += 1
                decomp_circ.append(gate.operation, qubit_id)

        direc = '../Data/decomp_circ_db/' + str(self.dim) + 'q/d' + str(d) + '_r' + str(n) + '/'
        # direc = 'Data/decomp_circ_db/benchmarks/circs/'
        # direc = 'Data/decomp_circ_db/random_trend/circs/'
        # filename = param + '.txt'
        filename = str(self.dim) + 'q_' + f'{param:03}' + '.txt'
        path = pathlib.Path(direc + filename)

        if save_qasm:
            qasm.dump(decomp_circ, path)

        if save_streams:
            path2 = '../Data/decomp_circ_db/' + str(self.dim) + 'q/d' + str(d) + '_r' + str(n) + '_streams/'
            # path2 = 'Data/decomp_circ_db/benchmarks/streams/'
            # path2 = 'Data/decomp_circ_db/random_trend/streams'
            file = str(self.dim) + 'q_' + f'{param:03}' + '.txt'
            # file = filename
            f = open(path2 + file, 'w')
            index = 0
            for inst in istream:
                if inst == "['cx']":
                    f.write(inst + ' ' + str(qstream[index]) + ' ' + str(qstream[index + 1]) + '\n')
                    index += 2
                else:
                    f.write(inst + ' ' + str(qstream[index]) + '\n')
                    index += 1

        return circ, decomp_circ, istream, qstream, tot_gateset, tot_sk_basis

    def skt_decomp(self, mat: np.ndarray, d: int, n: int, basis: list):
        skt = SolovayKitaevDecomposition(d, basic_approximations=basis)
        skt.__init__(d, basis)
        subcirc, dict_gateset, find_basics = skt.run(mat, n)
        basis_elements = [item.labels for item in basis]
        dict_basic = {str(obj): 0 for obj in basis_elements}
        help = Helper()
        help.__int__(dict_basic, find_basics, n)
        sk_basis = help.basic_traverse()
        instruction_stream = help.call_nested()
        del sk_basis['[]']
        return subcirc, instruction_stream, sk_basis

## Example usage

# dim = 2
# mat = np.array(random_unitary(2 ** dim), dtype=complex)
# dec = QuantumShannon_SKT()
# qsd = QuantumShannon_SKT()
# qsd.__int__(dim)
# d = 3
# n = 2
# gateset = ['h', 't']
# circ, decomp_circ, istream, qstream, tot_gateset, tot_sk_basis = qsd.decompose(mat,
#                                                                                d,
#                                                                                n,
#                                                                                gateset,
#                                                                                save_qasm=True,
#                                                                                save_streams=True,
#                                                                                param='trial')
# print(istream)
# print(qstream)