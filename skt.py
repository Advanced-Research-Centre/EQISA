import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from qiskit.synthesis import generate_basic_approximations
from qiskit.transpiler.passes import SolovayKitaev
import numpy as np

qc = QuantumCircuit(1)
qc.rz(-np.pi/3, 0)

print("Original circuit:")
print(qc.draw())

# Default: generate_basic_approximations(basis_gates=["h", "t", "tdg"], depth=10)
# Ref: 
# skd = SolovayKitaev(recursion_degree=2) 

# works only for _1q_gates = {"i","x","y","z","h","t","tdg","s","sdg","sx","sxdg"} as defined in import qiskit.circuit.library.standard_gates
# Ref: https://github.com/Qiskit/qiskit/blob/main/qiskit/synthesis/discrete_basis/generate_basis_approximations.py
# basis = ["s", "sdg", "t", "tdg", "z", "h"]
basis = ["t", "tdg", "h"]
gbs1 = generate_basic_approximations(basis, depth=1)
gbs2 = generate_basic_approximations(basis, depth=2)
gbs3 = generate_basic_approximations(basis, depth=3)
gbs4 = generate_basic_approximations(basis, depth=4)

# print(len(approx))
# print(type(approx[12]))
# print(approx[12].labels)
# print(approx[12].gates)
# print(approx[12].matrices)

for i in gbs1:
    print(i.labels,end=", ")
print()
# for i in gbs2:
#     print(i.labels,end=", ")
# print()
# for i in gbs3:
#     print(i.labels,end=", ")
# print()
for i in gbs4:
    print(i.labels,end=", ")
print()

# for existing in existing_sequences:
#   if matrix_equal(existing.product_su2, candidate.product_su2, ignore_phase=True, atol=tol):
#       ignore that candidate

skd1 = SolovayKitaev(recursion_degree=5, basic_approximations=gbs1)
skd2 = SolovayKitaev(recursion_degree=2, basic_approximations=gbs2)
skd3 = SolovayKitaev(recursion_degree=2, basic_approximations=gbs3)
skd4 = SolovayKitaev(recursion_degree=2, basic_approximations=gbs4)

discretized = skd1(qc)
print("Discretized circuit:")
print(discretized.draw())

discretized = skd2(qc)
print("Discretized circuit:")
print(discretized.draw())

discretized = skd3(qc)
print("Discretized circuit:")
print(discretized.draw())

discretized = skd4(qc)
print("Discretized circuit:")
print(discretized.draw())

# Ref: https://github.com/Qiskit/qiskit/blob/main/qiskit/synthesis/discrete_basis/solovay_kitaev.py
# Ref: https://github.com/Qiskit/qiskit/blob/main/qiskit/synthesis/discrete_basis/commutator_decompose.py

# if n == 0:
#     return self.find_basic_approximation(sequence)  
#     Args:       sequence: The gate to find the approximation to.
#     Returns:    Gate in basic approximations that is closest to ``sequence``.

# u_n1 = self._recurse(sequence, n - 1, check_input=check_input)

# v_n, w_n = commutator_decompose(
#     sequence.dot(u_n1.adjoint()).product, check_input=check_input
# )

# v_n1 = self._recurse(v_n, n - 1, check_input=check_input)
# w_n1 = self._recurse(w_n, n - 1, check_input=check_input)
# return v_n1.dot(w_n1).dot(v_n1.adjoint()).dot(w_n1.adjoint()).dot(u_n1)

# U.Uappx_dg = Uerr
# Uerr.Uappx = U.Uappx_dg.Uappx = U
# Uerr = V_dg.W_dg.V.W
# Vappx.Wappx.Vappx_dg.Wappx_dg.Uappx = Uappx2

# U.Uappx2_dg = Uerr2




 # simplify after skt
# _remove_identities(decomposition)
# _remove_inverse_follows_gate(decomposition)
