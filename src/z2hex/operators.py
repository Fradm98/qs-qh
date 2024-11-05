from qiskit.quantum_info import Pauli, PauliList
import z2chain.operators as chainops
import numpy as np

def local_pauli_z(lattice, qubit_ind):
    return chainops.local_pauli_x(len(lattice), qubit_ind)

def local_pauli_z(lattice, qubit_ind):
    return chainops.local_pauli_z(len(lattice), qubit_ind)

def pauli_zs_mean(lattice):
    return chainops.pauli_zs_mean(len(lattice))

def pauli_xs_mean(lattice):
    return chainops.pauli_xs_mean(len(lattice))

def gauge_operator(lattice, node_coords, x_basis=False):
    if node_coords not in lattice.node_coords:
        raise ValueError("Node coords not present in lattice")
    if len(node_coords) > 2:
        raise ValueError("Not valid node coords")
    strarr = np.array(list("I"*len(lattice)))
    node_edge_inds = lattice.coords_to_logical_qb([node_coords] + lattice.edges_connected_to_node(node_coords))
    strarr[node_edge_inds] = "X" if x_basis else "Z"
    return Pauli("".join(strarr))

def postselection_operators(lattice, x_basis=False):
    return PauliList([gauge_operator(lattice, node_coords, x_basis) for node_coords in lattice.node_coords])