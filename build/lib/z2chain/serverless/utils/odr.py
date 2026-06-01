import numpy as np

def compute_depolarizing_probability(measured_observable_arr, expected_observable_arr, observable_trace, nqubits):
    # PRE 104, 035309 (2021)
    try:
        return (np.array(measured_observable_arr) - np.array(expected_observable_arr))/(observable_trace/2**nqubits - np.array(expected_observable_arr))
    except ValueError:
        raise ValueError("measured_observable_arr expected_observable_arr and observable_trace must be broadcastable")
    
def compute_depolarizing_probability_std(measured_observable_std_arr, expected_observable_arr, observable_trace, nqubits):
    try:
        return np.abs(measured_observable_std_arr/(observable_trace/2**nqubits - np.array(expected_observable_arr)))
    except ValueError:
        raise ValueError("measured_observable_arr expected_observable_arr and observable_trace must be broadcastable")
    
def renormalize_observable(measured_observable_arr, p_arr, observable_trace, nqubits, observable_range=None, p_std=None, nantol=0.2):
    try:
        one_mask = np.isclose(p_arr, 1)
        p_arr[one_mask] = np.nan
        candidate = (np.array(measured_observable_arr) - np.array(p_arr)*observable_trace/2**nqubits) / (1 - np.array(p_arr))
        if p_std is not None: candidate_std = np.abs((np.array(measured_observable_arr) - np.array(p_arr)*observable_trace/2**nqubits)/(1 - np.array(p_arr))**2 - observable_trace/2**nqubits/(1-np.array(p_arr)))*p_std
        if observable_range is not None:
            mask = np.logical_or(candidate < (1+nantol)*observable_range[0] - nantol, candidate > (1+nantol)*observable_range[1] + nantol)
            candidate[mask] = np.nan
            if p_std is not None:
                candidate_std[mask] = np.nan
                return candidate, candidate_std
            else:
                return candidate
        else:
            if p_std is not None:
                return candidate, candidate_std
            else:
                return candidate
    except ValueError:
        raise ValueError("measured_observable_arr expected_observable_arr and observable_trace must be broadcastable")