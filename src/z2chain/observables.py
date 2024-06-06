from qiskit.quantum_info import SparsePauliOp

def local_pauli_z(chain_lenght, qubit_ind):
    nqubits = 2*chain_lenght - 1
    return SparsePauliOp("I"*qubit_ind + "Z" + "I"*(nqubits - qubit_ind - 1))