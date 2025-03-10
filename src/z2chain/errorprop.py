from qiskit_addon_utils.slicing import combine_slices, slice_by_gate_types
from z2chain.circs import string_quench_simulation_circuits_dt
from z2chain.operators import local_pauli_x, local_pauli_z
from qiskit_addon_utils.slicing import slice_by_barriers
from qiskit.quantum_info import SparsePauliOp
from utils.postselection import pauli_to_str
from qiskit_addon_obp import backpropagate
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt 
from itertools import product
import numpy as np

def get_error_slice(trotter_slices, error_layer_ind, error_inlayer_depth_pos, barriers=True):
    trotter_layer_depth = trotter_slices[0].depth()
    num_qbits = trotter_slices[0].num_qubits
    for slice in trotter_slices[1::]:
        if slice.depth() != trotter_layer_depth:
            raise ValueError("All slices must have the same depth")
        if slice.num_qubits != num_qbits:
            raise ValueError("All slices must have the same number of qubits")
    
    trotter_slices = trotter_slices.copy()
    slice_to_cut = trotter_slices[error_layer_ind]
    sliced_trotter_slice = slice_by_gate_types(slice_to_cut)
    right_slice = sliced_trotter_slice[error_inlayer_depth_pos::]

    if len(right_slice) == 0:
        return QuantumCircuit(trotter_slices[0].num_qubits)
    else:
        right_slice_combined = combine_slices(right_slice)
        error_slices_to_combine = [right_slice_combined] + trotter_slices[error_layer_ind + 1::]
        return combine_slices(error_slices_to_combine, include_barriers=True)
    
def propagate_error(slice, error_args):
    if not isinstance(error_args, tuple):
        raise ValueError("error args must be a tuple(int, int, int, Literal['X', 'Z'])")
    if len(error_args) != 4:
        raise ValueError("error args must be a tuple(int, int, int, Literal['X', 'Z'])")
    if not isinstance(slice, QuantumCircuit):
        raise ValueError("slice must be a QuantumCircuit")
    
    error_obs_func = {
        "X": local_pauli_x,
        "Z": local_pauli_z
    }
    error = SparsePauliOp(error_obs_func[error_args[-1]](slice.num_qubits, int(error_args[2])))
    propagated_error, remaining_slices, metadata = backpropagate(error, [slice])
    return propagated_error

def get_error_probabilities(propagated_error):
    propagated_error_flips = np.array([[1 if opel in {'X', 'Y'} else 0 for opel in pauli_to_str(operator)] for operator in propagated_error.paulis])
    propagated_error_phase = np.array([[1 if opel in {'Z', 'Y'} else 0 for opel in pauli_to_str(operator)] for operator in propagated_error.paulis])
    propagated_error_probs = np.abs(propagated_error.coeffs)**2
    error_flip_probabilities = np.sum(propagated_error_flips * propagated_error_probs.reshape((propagated_error_flips.shape[0], 1)), axis=0)
    error_phase_probabilities = np.sum(propagated_error_phase * propagated_error_probs.reshape((propagated_error_flips.shape[0], 1)), axis=0)
    return error_flip_probabilities, error_phase_probabilities

def get_all_error_args(trotter_slices, steps):
    error_args = []
    steps = len(trotter_slices)
    layer_len = trotter_slices[0].depth()
    nqubits = trotter_slices[0].num_qubits
    for i_steps in range(steps):
        for i_inlayer in range(layer_len + (i_steps == (steps - 1))):
            for i_qbit in range(nqubits):
                for error_type in ["X", "Z"]:
                    error_args.append((i_steps, i_inlayer, i_qbit, error_type))
    return error_args

def get_qubit_error_distribution(layer_slices, error_args):
    num_qbits = layer_slices[0].num_qubits
    flip_probabilities = np.zeros(num_qbits)
    phase_probabilities = np.zeros(num_qbits)
    num_errors = len(error_args)
    error_args = np.array(error_args, dtype=object)
    if error_args.shape[1] != 4:
        raise ValueError("error_args must have shape (n, 4)")
    error_args = sorted(error_args, key=lambda x: tuple(el for el in x))
    last_slice_args = tuple(error_args[0][:2])
    last_slice = get_error_slice(layer_slices, *last_slice_args, barriers=True)
    for eargs in error_args:
        this_slice_args = tuple(eargs[:2])
        if this_slice_args != last_slice_args:
            last_slice = get_error_slice(layer_slices, *this_slice_args, barriers=True)
            last_slice_args = this_slice_args
        propagated_error = propagate_error(last_slice, tuple(eargs))
        this_flip_probibilities, this_phase_probabilities = get_error_probabilities(propagated_error)
        flip_probabilities += this_flip_probibilities#/num_errors
        phase_probabilities += this_phase_probabilities#/num_errors
    
    return flip_probabilities, phase_probabilities

def pauli_string_length(pauli):
    pauli_str = pauli_to_str(pauli)
    pauli_chars = np.array(list(pauli_str))
    return np.sum(~np.equal(pauli_chars, 'I'))

def is_relevant_string(pauli, relevant_pauli_char):
    if relevant_pauli_char not in ["X", "Y", "Z"]:
        raise ValueError("relevant_pauli_char must be 'X', 'Y' or 'Z'")
    pauli_str = pauli_to_str(pauli)
    pauli_str_unique_chars = set(pauli_str)
    return pauli_str_unique_chars == {"I", relevant_pauli_char}

def get_prop_obs_pauli_distribution(layer_slices, observable, relevant_pauli_char=None):
    propagated_observable, remaining_slices, metadata = backpropagate(observable, layer_slices)
    all_string_lengths = np.array([pauli_string_length(pauli) for pauli in propagated_observable.paulis])
    all_string_weights = np.abs(propagated_observable.coeffs)**2

    # All strings
    unique_string_lengths = np.unique(all_string_lengths)
    unique_string_length_weights = np.zeros_like(unique_string_lengths, dtype=float)
    for i, length in enumerate(unique_string_lengths):
        unique_string_length_weights[i] = np.sum(all_string_weights[np.equal(all_string_lengths, length)])

    # Relevant strings
    if relevant_pauli_char is not None:
        relevant_string_mask = np.array([is_relevant_string(pauli, relevant_pauli_char) for pauli in propagated_observable.paulis])
        relevant_string_lengths = all_string_lengths[relevant_string_mask]
        relevant_string_weights = all_string_weights[relevant_string_mask]
        unique_relevant_string_lengths = np.unique(relevant_string_lengths)
        unique_relevant_string_length_weights = np.zeros_like(unique_relevant_string_lengths, dtype=float)
        for i, length in enumerate(unique_relevant_string_lengths):
            unique_relevant_string_length_weights[i] = np.sum(relevant_string_weights[np.equal(relevant_string_lengths, length)])
        return [unique_string_lengths, unique_string_length_weights], [unique_relevant_string_lengths, unique_relevant_string_length_weights]
    else:
        return [unique_string_lengths, unique_string_length_weights]
    
def plot_prop_obs_pauli_distribution(distribution):
    fig, ax = plt.subplots(figsize=[9, 6])
    pass