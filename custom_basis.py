# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

########################################################################################################################
# Code designed for the Energy-optimised QISA using Concept discovery and algorithmic complexity
# Custom-designed generate_basis_approximation method inspired from
# qiskit.synthesis.discrete_basis.generate_basis_approximations
# Also, uses other qiskit dependencies
########################################################################################################################
## UTILITIES
import collections
import numpy as np
import scipy

## QISKIT DEPENDENCIES
from qiskit.circuit.gate import Gate
from qiskit.quantum_info import random_unitary
# from qiskit.extensions import UnitaryGate
from qiskit.circuit.library import UnitaryGate
from qiskit.utils import optionals
from qiskit.synthesis.discrete_basis.gate_sequence import GateSequence
from qiskit.synthesis.discrete_basis.commutator_decompose import commutator_decompose


class UGate(Gate):
    U = np.identity(2)
    label = ""

    def __init__(self, label, unitary):
        self.U = unitary
        self.label = label
        """Create new gate."""
        super().__init__(label, 1, [], label=label)

    def inverse(self):
        """Invert this gate."""
        return UdgGate(self.label + 'dg', self.U)  # self-inverse

    def __array__(self, dtype=None):
        """Return a numpy.array for the U gate."""
        return self.U


class UdgGate(Gate):
    U = np.identity(2)
    label = ""

    def __init__(self, label, unitary):
        self.U = unitary
        self.label = label
        """ Create new gate. """
        super().__init__(label, 1, [], label=label)

    def inverse(self):
        """Invert this gate."""
        return UGate(self.label[:-2], self.U)  # self-inverse

    def __array__(self, dtype=None):
        """ Return a numpy.array for the Udg gate. """
        return scipy.linalg.inv(self.U)


Node = collections.namedtuple("Node", ("labels", "sequence", "children"))
_1q_gates = {}
_1q_inverses = {}

def _check_candidate_kdtree(candidate, existing_sequences, tol=1e-10):
    """Check if there's a candidate implementing the same matrix up to ``tol``.

    This uses a k-d tree search and is much faster than the greedy, list-based search.
    """
    from sklearn.neighbors import KDTree

    # do a quick, string-based check if the same sequence already exists
    if any(candidate.name == existing.name for existing in existing_sequences):
        return False

    points = np.array([sequence.product.flatten() for sequence in existing_sequences])
    candidate = np.array([candidate.product.flatten()])

    kdtree = KDTree(points)
    dist, _ = kdtree.query(candidate)

    return dist[0][0] > tol

def _process_node(node: Node, basis: list[str], sequences: list[GateSequence]):
    # inverse_last = _1q_inverses[node.labels[-1]] if node.labels else None

    for label in basis:
        # if label == inverse_last:
        #     continue

        sequence = node.sequence.copy()
        sequence.append(_1q_gates[label])

        if _check_candidate_kdtree(sequence, sequences):
            sequences.append(sequence)
            node.children.append(Node(node.labels + (label,), sequence, []))

    return node.children

def generate_basic_approximations(
                                  basis_gates: list[str | Gate], depth: int, filename: str | None = None
                                  ) -> list[GateSequence]:
    """Generates a list of ``GateSequence``s with the gates in ``basic_gates``.

    Args:
        basis_gates: The gates from which to create the sequences of gates.
        depth: The maximum depth of the approximations.
        filename: If provided, the basic approximations are stored in this file.

    Returns:
        List of ``GateSequences`` using the gates in ``basic_gates``.

    Raises:
        ValueError: If ``basis_gates`` contains an invalid gate identifier.
    """
    basis = []
    for gate in basis_gates:
        basis.append(gate.name)
        _1q_gates[gate.label] = gate

    for gate in basis:
        if gate.endswith('dg'):
            _1q_inverses[gate] = gate.removesuffix('dg')
            _1q_inverses[gate.removesuffix('dg')] = gate
        else:
            _1q_inverses[gate] = gate

    tree = Node((), GateSequence(), [])
    cur_level = [tree]
    sequences = [tree.sequence]
    for _ in [None] * depth:
        next_level = []
        for node in cur_level:
            next_level.extend(_process_node(node, basis, sequences))
        cur_level = next_level

    if filename is not None:
        data = {}
        for sequence in sequences:
            gatestring = sequence.name
            data[gatestring] = sequence.product

        np.save(filename, data)

    return sequences


def matrixCompare(A, B):
    if A.shape != B.shape:
        return False
    if A.dtype != B.dtype:
        return False
    return np.allclose(A, B, atol=10**-3)


def define_gateset(gate_seq):
    gs={}
    gtst = {}
    for gate in gate_seq:
        match gate:
            case 'i':       # Identity
                U = np.array([[1, 0], [0, 1]], dtype=complex)
            case 'x':       # Pauli X-gate: Rotation by pi radians around x-axis
                U = np.array([[0, 1], [1, 0]], dtype=complex)
            case 'y':       # Pauli Y-gate: Rotation by pi radians around y-axis
                U = np.array([[0, -1j], [1j, 0]], dtype=complex)
            case 'z':       # Pauli Z-gate: Rotation by pi radians around z-axis
                U = np.array([[1, 0], [0, -1]], dtype=complex)
            case 'h':       # Hadamard gate
                U = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]], dtype=complex)
            case 't':       # T gate: phase shift by pi/4 phase
                U = np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]], dtype=complex)
            case 'tdg':     # T dagger: phase shift by -pi/4
                U = np.array([[1, 0], [0, (1-1j)/np.sqrt(2)]], dtype=complex)
            case 's':       # S gate: phase shift by pi/2 phase
                U = np.array([[1, 0], [0, np.exp(1j*np.pi/2)]], dtype=complex)
            case 'sdg':     # S dagger: phase shift by-pi/2 phase
                U = np.array([[1, 0], [0, np.exp(-1j*np.pi/2)]], dtype=complex)
            case 'sx':       # Phase gate
                U = np.array([[1, 0], [0, 1j]], dtype=complex)
            case 'sxdg':
                U = np.array([[1, 0], [0, -1j]], dtype=complex)
            case 'rnd1':
                U = np.array(random_unitary(2), dtype=complex)
                # U = np.array([[0.39041259+0.50701289j, 0.76840783+0.00808385j],
                #               [0.12246614+0.75862901j, -0.5595164-0.3105245j]], dtype=complex)
            case 'rnd2':
                U = np.array(random_unitary(2), dtype=complex)
                # U = np.array([[-0.37368194-0.05487209j, 0.80362251-0.45993665j],
                #               [-0.6884152+0.61922159j, -0.06488061+0.37207478j]], dtype=complex)
            case 'rnd3':
                U = np.array(random_unitary(2), dtype=complex)
                # U = np.array([[-0.19189903-0.6662916j, -0.6723752-0.25911744j],
                #               [0.09379103+0.71444629j, -0.60250642-0.34315558j]], dtype=complex)
            case 'p1a':
                # from 200 Harr Random Unitaries
                U = np.array([[-0.78687181 - 5.68121785e-10j, -0.04577325 - 6.15416573e-01j],
                              [0.03316088 - 6.16224882e-01j, -0.7867068 + 1.61139095e-02j]], dtype=complex)
                # from 500 Harr Random Unitaries
                # U = np.array([[-0.99891178+9.50292836e-11j, -0.01683013-4.34972301e-02j],
                #                      [-0.0406026+2.29497486e-02j, 0.7722138+6.33648623e-01j]], dtype=complex)
            case 'p1b':
                # from 200 Harr Random Unitaries
                U = np.array([[0.8915669 - 1.45226897e-10j, 0.22308591 - 3.94133409e-01j],
                              [0.24090997 - 3.83498180e-01j, 0.42340431 + 7.84614761e-01j]], dtype=complex)
                # from 500 Harr Random Unitaries
                # U = np.array([[-0.54683177 + 0j, -0.52979855 - 0.64829662j],
                #               [0.8231603 + 0.15291220j, 0.26287606 + 0.47950095j]], dtype=complex)
            case _:
                print("Invalid Gate Set Configuration")
                exit()

        if bool(matrixCompare(U, np.matrix.getH(U))):
            # print("It's working!")
            gs[gate] = UnitaryGate(U, label=str(gate))
        else:
            gs[gate] = UnitaryGate(U, label=str(gate))
            gs[gate + 'dg'] = UnitaryGate(np.linalg.inv(U), label=str(gate) + 'dg')
        gtst[gate] = UnitaryGate(U, label=str(gate))
    gs_gates = ','.join(list(gs.keys()))

    return gs, gs_gates, gtst


def skt_gs(gs):

    gs_skt = []
    for g in gs.keys():
        if gs[g].num_qubits == 1:
            gs_skt.append(UGate(g, gs[g].to_matrix()))
    return gs_skt


def prep_gateset(gateset, depth):
    gs, gs_gates, gtst = define_gateset(gateset)
    return generate_basic_approximations(basis_gates=skt_gs(gs), depth=depth)


def identify_gates(gateset):
    _1q_gate = []
    _2q_gate = []
    for gate in gateset:
        match gate:
            case 'cx':
                _2q_gate.append(gate)
            case 'cz':
                _2q_gate.append(gate)
            case 'swap':
                _2q_gate.append(gate)
            case _:
                _1q_gate.append(gate)

    return _1q_gate, _2q_gate
