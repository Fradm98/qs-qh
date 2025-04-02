from utils.operations import expectation
import scipy.sparse as sparse
from utils import paulis
import numpy as np
import os

def sparse_hamiltonian_single_qubit_terms(lattice, J, h):
    sum_of_terms = sparse.csc_array((2**len(lattice), 2**len(lattice)), dtype=complex)

    for qbind in lattice.node_qbinds:
        sum_of_terms += -J * paulis.sparse_pauli_z(qbind, len(lattice))
    for qbind in lattice.edge_qbinds:
        sum_of_terms += -h * paulis.sparse_pauli_z(qbind, len(lattice))

    return sum_of_terms

def sparse_hamiltonian_interaction_term(lattice, lamb):
    sum_of_terms = sparse.csc_array((2**len(lattice), 2**len(lattice)), dtype=complex)

    for edge_coords in lattice.edge_coords:
        connected_nodes = lattice.nodes_connected_to_edge(edge_coords)
        interaction_qbinds = np.sort(lattice.coords_to_logical_qb([edge_coords, *connected_nodes]))
        sum_of_terms += -lamb * (paulis.sparse_pauli_x(interaction_qbinds[0], len(lattice)) @ 
                                 paulis.sparse_pauli_x(interaction_qbinds[1], len(lattice)) @
                                 paulis.sparse_pauli_x(interaction_qbinds[2], len(lattice)))
    return sum_of_terms

def sparse_hamiltonian(lattice, J, h, lamb):
    return sparse_hamiltonian_single_qubit_terms(lattice, J, h) + sparse_hamiltonian_interaction_term(lattice, lamb)

def sparse_single_qubit_propagator(lattice, J, h, t):
    return sparse.linalg.expm(-1j * t * sparse_hamiltonian_single_qubit_terms(lattice, J, h))

def sparse_interaction_propagator(lattice, lamb, t):
    return sparse.linalg.expm(-1j * t * sparse_hamiltonian_interaction_term(lattice, lamb))

def sparse_propagator(lattice, J, h, lamb, t):
    return sparse.linalg.expm(-1j * t * sparse_hamiltonian(lattice, J, h, lamb))

def sparse_2trotter_layer(lattice, J, h, lamb, t):
    sq_propagator = sparse_single_qubit_propagator(lattice, J, h, t / 2)
    int_propagator = sparse_interaction_propagator(lattice, lamb, t)
    return sq_propagator @ int_propagator @ sq_propagator

def string_initial_state(lattice, string_qubits):
    initial_state = np.zeros(2**len(lattice), dtype=complex)
    index = np.sum(2**(np.array(string_qubits)))
    initial_state[index] = 1
    return initial_state

def string_quench_exact_simulation(lattice, J, h, lamb, string_qubits, dt, steps, filepath="", overwrite=False, print_mode=False):
    if os.path.exists(filepath) and not overwrite:
        return np.loadtxt(filepath)
    
    initial_state = string_initial_state(lattice, string_qubits)
    pauliz_observables = [paulis.sparse_pauli_z(qbind, len(lattice)) for qbind in range(len(lattice))]
    base_propagator = sparse_propagator(lattice, J, h, lamb, dt)

    observable_matrix = np.zeros((steps+1, len(lattice)))
    current_state = initial_state.copy()
    observable_matrix[0] = np.array([expectation(current_state, obs) for obs in pauliz_observables])
    for i in range(steps):
        if print_mode:
            t = dt*i
            print(f"\rt = {t:.04f}".ljust(50), end="")
        current_state = base_propagator @ current_state
        observable_matrix[i+1] = np.array([expectation(current_state, obs) for obs in pauliz_observables])
    
    if filepath:
        np.savetxt(filepath, observable_matrix)

    return observable_matrix

def string_quench_trotter2_simulation(lattice, J, h, lamb, string_qubits, steps, dt, filepath="", overwrite=False, print_mode=False):
    if os.path.exists(filepath) and not overwrite:
        return np.loadtxt(filepath)
    
    initial_state = string_initial_state(lattice, string_qubits)
    pauliz_observables = [paulis.sparse_pauli_z(qbind, len(lattice)) for qbind in range(len(lattice))]
    base_propagator = sparse_2trotter_layer(lattice, J, h, lamb, dt)

    observable_matrix = np.zeros((steps+1, len(lattice)))
    current_state = initial_state.copy()
    observable_matrix[0] = np.array([expectation(current_state, obs) for obs in pauliz_observables])
    for i in range(steps):
        if print_mode:
            t = dt*i
            print(f"\rt = {t:.04f}".ljust(50), end="")
        current_state = base_propagator @ current_state
        observable_matrix[i+1] = np.array([expectation(current_state, obs) for obs in pauliz_observables])
    
    if filepath:
        np.savetxt(filepath, observable_matrix)
    
    return observable_matrix

def string_quench_trotter2_simulation_final_time(lattice, J, h, lamb, string_qubits, t_final, steps, dt, filepath="", overwrite=False, print_mode=False):
    if os.path.exists(filepath) and not overwrite:
        return np.loadtxt(filepath)
    
    initial_state = string_initial_state(lattice, string_qubits)
    pauliz_observables = [paulis.sparse_pauli_z(qbind, len(lattice)) for qbind in range(len(lattice))]

    t_arr = np.linspace(0, t_final, steps+1)
    nlayers_arr = np.ceil(t_arr[1::] / dt).astype(int)
    dt_arr = t_arr[1::] / nlayers_arr

    observable_matrix = np.zeros((steps+1, len(lattice)))
    current_state = initial_state.copy()
    observable_matrix[0] = np.array([expectation(current_state, obs).real for obs in pauliz_observables])
    for i in range(steps):
        if print_mode:
            t = t_arr[i]
            print(f"\rt = {t:.04f}".ljust(50), end="")
        this_t_state = initial_state.copy()
        base_propagator = sparse_2trotter_layer(lattice, J, h, lamb, dt_arr[i])
        for j in range(nlayers_arr[i]):
            this_t_state = base_propagator @ this_t_state
        observable_matrix[i+1] = np.array([expectation(this_t_state, obs).real for obs in pauliz_observables])

    if filepath:
        np.savetxt(filepath, observable_matrix)

    return observable_matrix      