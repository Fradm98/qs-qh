from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp

def local_pauli_z(nqubits, qubit_ind):
    return Pauli("I"*qubit_ind + "Z" + "I"*(nqubits - qubit_ind - 1))

def local_pauli_x(nqubits, qubit_ind):
    return Pauli("I"*qubit_ind + "X" + "I"*(nqubits - qubit_ind - 1))

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

def gauge_operator(chain_length, site_ind, x_basis=False):
    if not site_ind < chain_length:
        raise ValueError("Arguments must fulfill chain_length < site_ind")
    base_str = "XXX" if x_basis else "ZZZ"
    if site_ind == 0:
        paulistr = base_str[:-1] + "I"*(2*(chain_length-1)-1)
    elif site_ind == (chain_length - 1):
        paulistr = "I"*(2*(chain_length-1)-1) + base_str[:-1]
    else:
        paulistr = "I"*(2*site_ind - 1) + base_str + "I"*(2*(chain_length - site_ind - 1) - 1)
    return Pauli(paulistr)