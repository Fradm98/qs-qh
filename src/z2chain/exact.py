from utils.operations import expectation
import utils.multiproductf as mpf
import scipy.sparse as sparse
from utils import paulis
import numpy as np
import os

def sparse_hamiltonian_single_qubit_terms(J, h, chain_length):
    nqubits = 2*chain_length - 1
    sum_of_terms = sparse.csc_array((2**nqubits, 2**nqubits), dtype=complex)

    for n in range(chain_length - 1):
        sum_of_terms += -J*paulis.sparse_pauli_z(2*n, nqubits) - h*paulis.sparse_pauli_z(2*n + 1, nqubits)
    sum_of_terms += -J*paulis.sparse_pauli_z(nqubits - 1, nqubits)

    return sum_of_terms

def sparse_single_qubit_terms_propagator(J, h, chain_length, t):
    sq_terms = sparse_hamiltonian_single_qubit_terms(J, h, chain_length)
    return sparse.linalg.expm(-1j*sq_terms*t)

def sparse_hamiltonian_interaction_terms(lamb, chain_length):
    nqubits = 2*chain_length - 1
    sum_of_terms = sparse.csc_array((2**nqubits, 2**nqubits), dtype=complex)

    for n in range(chain_length - 1):
        sum_of_terms += -lamb*(paulis.sparse_pauli_x(2*(n+1), nqubits) @ paulis.sparse_pauli_x(2*n + 1, nqubits) @ paulis.sparse_pauli_x(2*n, nqubits))

    return sum_of_terms

def sparse_interaction_propagator(lamb, chain_length, t):
    interaction_terms = sparse_hamiltonian_interaction_terms(lamb, chain_length)
    return sparse.linalg.expm(-1j*interaction_terms*t)

def sparse_diagonal_propagator(J, h, chain_length, t):
    diagonal_terms = sparse_hamiltonian_single_qubit_terms(J, h, chain_length)
    return sparse.linalg.expm(-1j*diagonal_terms*t)

def sparse_qubit_occupation_operator(qubit, chain_length):
    nqubits = 2*chain_length - 1
    return (sparse.identity(2**nqubits) - paulis.sparse_pauli_z(qubit, nqubits)) / 2

def sparse_hamiltonian(J, h, lamb, chain_length):
    nqubits = 2*chain_length - 1
    hamiltonian = sparse.csc_array((2**nqubits, 2**nqubits), dtype=complex)

    for n in range(chain_length - 1):
        hamiltonian += -J*paulis.sparse_pauli_z(2*n, nqubits) - h*paulis.sparse_pauli_z(2*n + 1, nqubits)
        hamiltonian += -lamb*(paulis.sparse_pauli_x(2*(n+1), nqubits) @ paulis.sparse_pauli_x(2*n + 1, nqubits) @ paulis.sparse_pauli_x(2*n, nqubits))
    hamiltonian += -J*paulis.sparse_pauli_z(2*chain_length - 2, nqubits)

    return hamiltonian

def sparse_propagator(J, h, lamb, chain_length, t):
    hamiltonian = sparse_hamiltonian(J, h, lamb, chain_length)
    return sparse.linalg.expm(-1j*hamiltonian*t)

def sparse_dual_site_occupation_operator(site, chain_length):
    nqubits = chain_length - 1
    if site == 0:
        pauli_part = paulis.sparse_pauli_z(0, nqubits)
    elif site == (chain_length - 1):
        pauli_part = paulis.sparse_pauli_z(nqubits - 1, nqubits)
    else:
        pauli_part = paulis.sparse_pauli_z(site - 1, nqubits) @ paulis.sparse_pauli_z(site, nqubits)
    return (sparse.identity(2**nqubits) - pauli_part)/2
    
def sparse_dual_gauge_occupation_operator(left_site, chain_length):
    nqubits = chain_length - 1
    return (sparse.identity(2**nqubits) - paulis.sparse_pauli_z(left_site, nqubits))/2

def sparse_dual_hamiltonian(J, h, lamb, chain_length):
    nqubits = chain_length - 1

    hamiltonian = -J*(paulis.sparse_pauli_z(0, nqubits) + paulis.sparse_pauli_z(nqubits - 1, nqubits))
    for n in range(nqubits):
        if n < nqubits - 1:
            hamiltonian += -J*(paulis.sparse_pauli_z(n, nqubits) @ paulis.sparse_pauli_z(n + 1, nqubits))
        hamiltonian += -h*paulis.sparse_pauli_z(n, nqubits)
        hamiltonian += -lamb*paulis.sparse_pauli_x(n, nqubits)

    return hamiltonian

def sparse_dual_propagator(J, h, lamb, chain_length, t):
    hamiltonian = sparse_dual_hamiltonian(J, h, lamb, chain_length)
    return sparse.linalg.expm(-1j*hamiltonian*t)

def particle_pair_initial_state(left_particle_position, chain_length):
    nqubits = 2*chain_length - 1
    state = np.zeros(2**nqubits)
    index = 2**(nqubits - 2*left_particle_position - 3) + 2**(nqubits - 2*left_particle_position - 2) + 2**(nqubits - 2*left_particle_position - 1)
    state[index] = 1
    return state

def singlep_initial_state(particle_position, chain_length):
    nqubits = 2*chain_length - 1
    state = np.zeros(2**nqubits)
    index = 2**(nqubits - 2*particle_position - 1) + np.sum(2**(nqubits - 2*particle_position - 2 - 2*np.arange(chain_length//2))) 
    state[index] = 1
    return state

def dual_particle_pair_initial_state(left_particle_position, chain_length):
    nqubits = chain_length - 1
    state = np.zeros(2**nqubits)
    state[2**(nqubits - left_particle_position - 1)] = 1
    return state

def particle_pair_quench_simulation(L, J, h, lamb, particle_pair_left_position, dt, steps, filepath="", overwrite=False, print_mode=False):    
    if os.path.exists(filepath) and not overwrite:
        return np.loadtxt(filepath)
    
    initial_state = particle_pair_initial_state(particle_pair_left_position, L)
    if print_mode: print("\rCreating site occupation operators...".ljust(50), end="")
    occupation_operators = [sparse_qubit_occupation_operator(i, L) for i in range(2*L - 1)]
    if print_mode: print("\rCreating propagator...".ljust(50), end="")
    base_propagator = sparse_propagator(J, h, lamb, L, dt)

    site_gauge_occupation_matrix = np.zeros((steps + 1, 2*L-1))
    current_state = initial_state.copy()
    for i in range(steps + 1):
        if print_mode:
            t = dt*i
            print(f"\rt = {t:.04f}".ljust(50), end="")
        this_qubits_occupation = np.array([expectation(current_state, socc_op).real for socc_op in occupation_operators])
        site_gauge_occupation_matrix[i] = this_qubits_occupation
        current_state = base_propagator @ current_state

    if filepath:
        header = f"L = {L} / J = {J} / lamb = {lamb}\nparticle_position = {particle_pair_left_position}\nfinal_time = {dt*steps:.04f} / steps = {steps}"
        np.savetxt(filepath, site_gauge_occupation_matrix, header=header)
    
    return site_gauge_occupation_matrix

def singlep_quench_simulation(L, J, h, lamb, particle_position, dt, steps, filepath="", overwrite=False, print_mode=False):
    if os.path.exists(filepath) and not overwrite:
        return np.loadtxt(filepath)

    initial_state = singlep_initial_state(particle_position, L)
    if print_mode: print("\rCreating occupation operators...".ljust(50), end="")
    occupation_operators = [paulis.sparse_pauli_z(i, 2*L-1) for i in range(2*L - 1)]
    if print_mode: print("\rCreating propagator...".ljust(50), end="")
    base_propagator = sparse_propagator(J, h, lamb, L, dt)

    site_gauge_occupation_matrix = np.zeros((steps + 1, 2*L-1))
    current_state = initial_state.copy()
    for i in range(steps + 1):
        if print_mode:
            t = dt*i
            print(f"\rt = {t:.04f}".ljust(50), end="")
        this_qubits_occupation = np.array([expectation(current_state, socc_op).real for socc_op in occupation_operators])
        site_gauge_occupation_matrix[i] = this_qubits_occupation
        current_state = base_propagator @ current_state

    if filepath:
        extension = os.path.splitext(filepath)[1]
        if extension == ".npy":
            np.save(filepath, site_gauge_occupation_matrix)
        else:
            header = f"L = {L} / J = {J} / lamb = {lamb}\nparticle_position = {particle_position}\nfinal_time = {dt*steps:.04f} / steps = {steps}"
            np.savetxt(filepath, site_gauge_occupation_matrix, header=header)
    return site_gauge_occupation_matrix

def singlep_quench_trotter2_simulation(L, J, h, lamb, particle_position, dt, steps, filepath="", overwrite=False):
    if os.path.exists(filepath) and not overwrite:
        return np.loadtxt(filepath)
    
    initial_state = singlep_initial_state(particle_position, L)
    diagonal_propagator = sparse_diagonal_propagator(J, h, L, dt/2)
    interaction_propagator = sparse_interaction_propagator(lamb, L, dt)
    step_propagator = diagonal_propagator @ interaction_propagator @ diagonal_propagator
    occupation_operators = [paulis.sparse_pauli_z(i, 2*L-1) for i in range(2*L - 1)]
    observable_matrix = np.zeros((steps + 1, 2*L - 1))
    observable_matrix[0] = np.array([expectation(initial_state, socc_op).real for socc_op in occupation_operators])
    current_state = initial_state.copy()
    for i in range(steps):
        current_state = step_propagator @ current_state
        observable_matrix[i + 1] = np.array([expectation(current_state, socc_op).real for socc_op in occupation_operators])

    if filepath:
        extension = os.path.splitext(filepath)[1]
        if extension == ".npy":
            np.save(filepath, observable_matrix)
        else:
            np.savetxt(filepath, observable_matrix)
    
    return observable_matrix

def singlep_quench_mpf2_simulation_single_t(L, J, h, lamb, particle_position, ks, coeffs, t):
    initial_state = singlep_initial_state(particle_position, L)
    occupation_operators = [paulis.sparse_pauli_z(i, 2*L-1) for i in range(2*L - 1)]
    observable_matrix = np.zeros(2*L - 1)
    for i, k in enumerate(ks):
        this_diagonal_propagator = sparse_diagonal_propagator(J, h, L, t/2/k)
        this_interaction_propagator = sparse_interaction_propagator(lamb, L, t/k)
        this_propagator = this_diagonal_propagator @ this_interaction_propagator @ this_diagonal_propagator
        this_propagator = sparse.linalg.matrix_power(this_propagator, k)
        this_final_state = this_propagator @ initial_state
        observable_matrix += coeffs[i] * np.array([expectation(this_final_state, socc_op).real for socc_op in occupation_operators])
    
    return observable_matrix

def singlep_quench_mpf2_simulation(L, J, h, lamb, particle_position, ks, coeffs, dt, filepath="", overwrite=False):
    if os.path.exists(filepath) and not overwrite:
        return np.loadtxt(filepath)
    
    ks = np.array(ks)
    coeffs = np.array(coeffs)
    if ks.shape != coeffs.shape:
        raise ValueError("ks and coeffs must have the same shape")
    
    if len(ks.shape) == 1:
        raise ValueError("ks and coeffs must be 2D matrices")
    steps = ks.shape[0]

    initial_state = singlep_initial_state(particle_position, L)
    occupation_operators = [paulis.sparse_pauli_z(i, 2*L-1) for i in range(2*L - 1)]
    observable_matrix = np.zeros((steps + 1, 2*L - 1))
    observable_matrix[0] = np.array([expectation(initial_state, socc_op).real for socc_op in occupation_operators])
    for i in range(steps):
        observable_matrix[i+1] = singlep_quench_mpf2_simulation_single_t(L, J, h, lamb, particle_position, ks[i], coeffs[i], dt*(i+1))
    
    if filepath:
        extension = os.path.splitext(filepath)[1]
        if extension == ".npy":
            np.save(filepath, observable_matrix)
        else:
            np.savetxt(filepath, observable_matrix)
    
    return observable_matrix
    
def trotter_particle_pair_quench_simulation(L, J, h, lamb, particle_pair_left_position, final_time, layers, measure_every_layers=1, filepath="", overwrite=False, print_mode=False):
    if os.path.exists(filepath) and not overwrite:
        return np.loadtxt(filepath)
    
    initial_state = particle_pair_initial_state(particle_pair_left_position, L)
    if print_mode: print("\rCreating occupation operators...".ljust(50), end="")
    occupation_operators = [sparse_qubit_occupation_operator(i, L) for i in range(2*L - 1)]
    if print_mode: print("\rCreating propagator...".ljust(50), end="")
    layer_time = final_time / layers
    sqterms_trotter_factor = sparse_single_qubit_terms_propagator(J, h, L, layer_time)
    interaction_trotter_factor = sparse_interaction_propagator(lamb, L, layer_time)

    site_gauge_occupation_matrix = np.zeros((layers // measure_every_layers + 1, 2*L-1))
    current_state = initial_state.copy()
    for i in range(layers):
        if print_mode:
            t = i*layers/final_time
            print(f"\rt = {t:.04f} / t_f = {final_time:.04f} / Trotter_steps = {i} of {layers}".ljust(50), end="")
        this_qubits_occupation = np.array([expectation(current_state, socc_op) for socc_op in occupation_operators])
        if i % measure_every_layers == 0:
            site_gauge_occupation_matrix[i // measure_every_layers] = this_qubits_occupation
        current_state = sqterms_trotter_factor @ interaction_trotter_factor @ current_state

    this_qubits_occupation = np.array([expectation(current_state, socc_op) for socc_op in occupation_operators])
    site_gauge_occupation_matrix[-1] = this_qubits_occupation
    
    if filepath:
        header = f"L = {L} / J = {J} / lamb = {lamb}\nparticle_position = {particle_pair_left_position}\nfinal_time = {final_time} / layers = {layers}"
        np.savetxt(filepath, site_gauge_occupation_matrix, header=header)

    return site_gauge_occupation_matrix

def dual_particle_pair_quench_simulation(L, J, h, lamb, particle_pair_left_position, final_time, steps, filepath="", overwrite=False, print_mode=False):    
    if os.path.exists(filepath) and not overwrite:
        return np.loadtxt(filepath)
    
    initial_state = dual_particle_pair_initial_state(particle_pair_left_position, L)
    if print_mode: print("\rCreating site occupation operators...".ljust(50), end="")
    site_occupation_operators = [sparse_dual_site_occupation_operator(n, L) for n in range(L)]
    if print_mode: print("\rCreating gauge occupation operators...".ljust(50), end="")
    gauge_occupation_operators = [sparse_dual_gauge_occupation_operator(n, L) for n in range(L-1)]
    if print_mode: print("\rCreating propagator...".ljust(50), end="")
    base_propagator = sparse_dual_propagator(J, h, lamb, L, final_time/steps)

    site_gauge_occupation_matrix = np.zeros((steps + 1, 2*L-1))
    current_state = initial_state.copy()
    for i in range(steps + 1):
        if print_mode:
            t = i*steps/final_time 
            print(f"\rt = {t:.04f} / t_f = {t:.04f}".ljust(50), end="")
        this_sites_occupation = np.array([expectation(current_state, socc_op).real for socc_op in site_occupation_operators])
        this_gauges_occupation = np.array([expectation(current_state, gocc_op).real for gocc_op in gauge_occupation_operators])
        site_gauge_occupation_matrix[i, ::2] = this_sites_occupation
        site_gauge_occupation_matrix[i, 1::2] = this_gauges_occupation
        current_state = base_propagator @ current_state

    if filepath:
        header = f"L = {L} / J = {J} / lamb = {lamb}\nparticle_position = {particle_pair_left_position}\nfinal_time = {final_time} / steps = {steps}"
        np.savetxt(filepath, site_gauge_occupation_matrix, header=header)
    
    return site_gauge_occupation_matrix

def dual_trotter_particle_pair_quench_simulation(L, J, h, lamb, particle_pair_left_position, final_time, layers, measure_every_layers=1, filepath="", overwrite=False, print_mode=False):
    raise NotImplementedError()