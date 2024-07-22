from qiskit.quantum_info import SparsePauliOp

def local_pauli_z(nqubits, qubit_ind):
    return SparsePauliOp("I"*qubit_ind + "Z" + "I"*(nqubits - qubit_ind - 1))

def local_pauli_x(nqubits, qubit_ind):
    return SparsePauliOp("I"*qubit_ind + "X" + "I"*(nqubits - qubit_ind - 1))