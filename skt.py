import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from qiskit.synthesis import generate_basic_approximations
from qiskit.transpiler.passes import SolovayKitaev
import numpy as np

qc = QuantumCircuit(1)
qc.rz(-np.pi/2, 0)

print("Original circuit:")
print(qc.draw())

# works only for default basis gate set of ["t", "tdg", "h"]
# skd = SolovayKitaev(recursion_degree=2) 

# works only for _1q_gates = {"i","x","y","z","h","t","tdg","s","sdg","sx","sxdg"} as defined in import qiskit.circuit.library.standard_gates
# Ref: https://github.com/Qiskit/qiskit/blob/main/qiskit/synthesis/discrete_basis/generate_basis_approximations.py
basis = ["s", "sdg", "t", "tdg", "z", "h"]
approx = generate_basic_approximations(basis, depth=2)


# print(len(approx))
# print(type(approx[12]))
# print(approx[12].labels)
# print(approx[12].gates)
# print(approx[12].matrices)

for i in approx:
    print(i.labels)

# for existing in existing_sequences:
#   if matrix_equal(existing.product_su2, candidate.product_su2, ignore_phase=True, atol=tol):
#       ignore that candidate

# skd = SolovayKitaev(recursion_degree=2, basic_approximations=approx)

# discretized = skd(qc)

# print("Discretized circuit:")
# print(discretized.draw())
