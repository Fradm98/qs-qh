import scipy.sparse as sparse
from utils import paulis

def sparse_hamiltonian(J, h, lamb, chain_length):
    nqubits = 2*chain_length - 1
    hamiltonian = sparse.csc_array((2**nqubits, 2**nqubits), dtype=complex)

    for n in range(chain_length - 1):
        hamiltonian += -J*paulis.sparse_pauli_z(2*n, nqubits) - h*paulis.sparse_pauli_x(2*n + 1, nqubits)
        hamiltonian += -lamb*(paulis.sparse_pauli_x(2*(n+1), nqubits) @ paulis.sparse_pauli_x(2*n + 1, nqubits) @ paulis.sparse_pauli_x(2*n, nqubits))
    hamiltonian += -paulis.sparse_pauli_z(chain_length)

    return hamiltonian

def sparse_propagator(J, h, lamb, chain_length, t):
    hamiltonian = sparse_hamiltonian(J, h, lamb, chain_length)
    return sparse.linalg.expm(-1j*hamiltonian*t)

def sparse_dual_hamiltonian(J, h, lamb, chain_length):
    nqubits = chain_length - 1

    hamiltonian = -J*(paulis.sparse_pauli_z(0, nqubits) + paulis.sparse_pauli_z(nqubits-1, nqubits))
    for n in range(chain_length - 1):
        if n < (nqubits - 1):
            hamiltonian += -J*(paulis.sparse_pauli_z(n, nqubits) @ paulis.sparse_pauli_z(n+1, nqubits))
        hamiltonian += -h*paulis.sparse_pauli_z(n, nqubits)
        hamiltonian += -lamb*paulis.sparse_pauli_x(n, nqubits)

    return hamiltonian

def sparse_dual_propagator(J, h, lamb, chain_length, t):
    hamiltonian = sparse_dual_hamiltonian(J, h, lamb, chain_length)
    return sparse.linalg.expm(-1j*hamiltonian*t)