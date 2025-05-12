from utils.odr import compute_depolarizing_probability, renormalize_observable
from utils.bootstrap import bootstrap_sample_dict_mean
from multiprocessing import Pool, cpu_count
from itertools import product
import numpy as np

def single_load_occupation_correlators(samples_dict, occupied_qubits, confidence=None, shots=None):
    if not isinstance(occupied_qubits, (list, tuple, np.ndarray)):
        raise ValueError("occupied_qubits must be a list, tuple, or numpy array")
    if len(samples_dict) == 0:
        if confidence is not None:
            return np.array(np.nan), np.array([[np.nan, np.nan]])
        else: 
            return np.array(np.nan)
    
    nqubits = len(list(samples_dict.keys())[0])
    string_projectors_basis_values = np.zeros(len(samples_dict), dtype=int)
    qubit_expectation_mask = np.array([qubit in occupied_qubits for qubit in range(nqubits)], dtype=bool)
    for i, state_string in enumerate(samples_dict.keys()):
        state_string_rep = np.array([int(c) for c in state_string], dtype=int)[::-1]
        string_projectors_basis_values[i] = int(np.sum(state_string_rep[qubit_expectation_mask]) == len(occupied_qubits))
    weights = np.array(list(samples_dict.values()))/np.sum(list(samples_dict.values()))
    projector_values = np.sum(string_projectors_basis_values * weights)
    if confidence is not None:
        if list(samples_dict.values())[0] < 1:
            if shots is None:
                raise ValueError("To boostrap a probability dictionary, shots must be provided")
            bootstrap_samples = {k:int(np.floor(v*shots)) for k, v in samples_dict.items()}
        else:
            bootstrap_samples = samples_dict
        confidence_intervals = bootstrap_sample_dict_mean(bootstrap_samples, string_projectors_basis_values.reshape((-1, 1)), confidence)
        return projector_values, confidence_intervals
    else:
        return projector_values
    
def load_projection_operators(samples_dicts, strings_qubits, cal_samples_dicts=None, confidence=None, shots=None):
    if isinstance(samples_dicts, dict):
        samples_dicts = [samples_dicts]
    elif not isinstance(samples_dicts, (list, tuple, np.ndarray)):
        raise ValueError("samples_dicts must be a list, tuple, or numpy array")
    samples_dicts = np.array(samples_dicts, dtype=object)
    if len(samples_dicts.shape) != 1:
        raise ValueError("samples_dicts must be a 1D array of dictionaries")
    if cal_samples_dicts is not None:
        cal_samples_dicts = np.array(cal_samples_dicts, dtype=object)
        if cal_samples_dicts.shape != samples_dicts.shape:
            raise ValueError("cal_samples_dicts must have the same shape as samples_dicts")
        
    projection_operators_values = np.zeros((len(samples_dicts), len(strings_qubits)), dtype=float)
    if confidence is not None:
        confidence_intervals = np.zeros((len(samples_dicts), len(strings_qubits), 2), dtype=float)
    results = []
    if cal_samples_dicts is not None:
        results_cal = []
    for i, samples_dict in enumerate(samples_dicts):
        for j, string_qubit in enumerate(strings_qubits):
            results.append(single_load_occupation_correlators(samples_dict, string_qubit, confidence=confidence, shots=shots))
            if cal_samples_dicts is not None:
                results_cal.append(single_load_occupation_correlators(cal_samples_dicts[i], string_qubit, confidence=confidence, shots=shots))
    # if confidence is not None:
    #     return projection_operators_values, confidence_intervals
    # else:
    #     return projection_operators_values
    
    # mp_arguments = [(samples, string, confidence, shots) for samples, string in product(samples_dicts, strings_qubits)]
    # if cal_samples_dicts is not None:
    #     mp_arguments_cal = [(samples, string, confidence, shots) for samples, string in product(cal_samples_dicts, strings_qubits)]
    # with Pool(processes=cpu_count()) as pool:
    #     results = pool.starmap(single_load_occupation_correlators, mp_arguments)
    #     if cal_samples_dicts is not None:
    #         results_cal = pool.starmap(single_load_occupation_correlators, mp_arguments_cal)
    
    if confidence is not None:
        projection_operators_values = np.array([result[0] for result in results]).reshape(len(samples_dicts), len(strings_qubits))
        confidence_intervals = np.array([result[1] for result in results]).reshape(len(samples_dicts), len(strings_qubits), 2)
        to_return = [[projection_operators_values], [confidence_intervals]]
        if cal_samples_dicts is not None:
            projection_operators_values_cal = np.array([result[0] for result in results_cal]).reshape(len(cal_samples_dicts), len(strings_qubits))
            confidence_intervals_cal = np.array([result[1] for result in results_cal]).reshape(len(cal_samples_dicts), len(strings_qubits), 2)
            to_return[0].append(projection_operators_values_cal)
            to_return[1].append(confidence_intervals_cal)
            nqubits = len(list(samples_dicts[0].keys())[0])
            # NOTE: Trace can change depending on the order of the correlator
            corrected_projection_operators_values = np.zeros_like(projection_operators_values)
            corrected_confidence_intervals = np.zeros_like(confidence_intervals)
            for i, corr_vals in enumerate(projection_operators_values.T):
                p_arr = compute_depolarizing_probability(projection_operators_values_cal[:, i], 0, 2**nqubits/2**len(strings_qubits[i]), nqubits)
                this_corrected_projection_operators_values, this_corrected_confidence_intervals = renormalize_observable(corr_vals, p_arr, 2**nqubits/2**len(strings_qubits[i]), nqubits, observable_confidence=confidence_intervals_cal[:, i, :], expected_observable_arr=0)
                corrected_projection_operators_values[:, i] = this_corrected_projection_operators_values
                corrected_confidence_intervals[:, i, :] = this_corrected_confidence_intervals
            to_return[0].insert(0, corrected_projection_operators_values)
            to_return[1].insert(0, corrected_confidence_intervals)
        return to_return
    else:
        projection_operators_values = np.array(results).reshape(len(samples_dicts), len(strings_qubits))
        to_return = [projection_operators_values]
        if cal_samples_dicts is not None:
            projection_operators_values_cal = np.array(results_cal).reshape(len(cal_samples_dicts), len(strings_qubits))
            to_return.append(projection_operators_values_cal)
        return to_return