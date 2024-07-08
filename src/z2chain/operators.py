from qiskit.quantum_info import SparsePauliOp

def local_pauli_z(nqubits, qubit_ind):
    return SparsePauliOp("I"*qubit_ind + "Z" + "I"*(nqubits - qubit_ind - 1))