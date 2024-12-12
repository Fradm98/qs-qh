from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp

def local_pauli_z(nqubits, qubit_ind):
    if qubit_ind > nqubits:
        raise ValueError("qubit_ind must be in the interval [0, nqubits)")
    return Pauli("I"*qubit_ind + "Z" + "I"*(nqubits - qubit_ind - 1))

def local_pauli_x(nqubits, qubit_ind):
    if qubit_ind > nqubits:
        raise ValueError("qubit_ind must be in the interval [0, nqubits)")
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

def postselection_operators(chain_length, x_basis=False):
    nqubits = 2*chain_length - 1
    basis_string = "X" if x_basis else "Z"
    postselection_ops = [Pauli("I"*(2*i - 1) + basis_string*3 + "I"*(2*(chain_length - i - 1) - 1)) for i in range(1, chain_length-1)]
    postselection_ops.insert(0, Pauli(basis_string*2 + "I"*(nqubits - 2)))
    postselection_ops.append(Pauli("I"*(nqubits - 2) + basis_string*2))
    return PauliList(postselection_ops)