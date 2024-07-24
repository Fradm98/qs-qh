from qiskit.quantum_info import SparsePauliOp

def local_pauli_z(nqubits, qubit_ind):
    return SparsePauliOp("I"*qubit_ind + "Z" + "I"*(nqubits - qubit_ind - 1))

def local_pauli_x(nqubits, qubit_ind):
    return SparsePauliOp("I"*qubit_ind + "X" + "I"*(nqubits - qubit_ind - 1))

def pauli_zs_mean(nqubits):
    paulistrs = ["I"*i + "Z" + "I"*(nqubits - 1 - i) for i in range(nqubits)]
    coeffs = 1 / nqubits
    logical_observable = SparsePauliOp(paulistrs, coeffs)
    return logical_observable

def pauli_xs_mean(nqubits):
    paulistrs = ["I"*i + "X" + "I"*(nqubits - 1 - i) for i in range(nqubits)]
    coeffs = 1 / nqubits
    logical_observable = SparsePauliOp(paulistrs, coeffs)
    return logical_observable