import numpy as np

def compute_depolarizing_probability(measured_observable_arr, expected_observable_arr, observable_trace, nqubits):
    # PRE 104, 035309 (2021)
    try:
        return (np.array(measured_observable_arr) - np.array(expected_observable_arr))/(observable_trace/2**nqubits - np.array(expected_observable_arr))
    except ValueError:
        raise ValueError("measured_observable_arr expected_observable_arr and observable_trace must be broadcastable")
    
def renormalize_observable(measured_observable_arr, p_arr, observable_trace, nqubits, observable_range=None, nantol=0.2):
    try:
        one_mask = np.isclose(p_arr, 1)
        p_arr[one_mask] = np.nan
        candidate = (np.array(measured_observable_arr) - np.array(p_arr)*observable_trace/2**nqubits) / (1 - np.array(p_arr))
        if observable_range is not None:
            mask = np.logical_or(candidate < (1+nantol)*observable_range[0], candidate > (1+nantol)*observable_range[1])
            candidate[mask] = np.nan
            return candidate
        else:
            return candidate
    except ValueError:
        raise ValueError("measured_observable_arr expected_observable_arr and observable_trace must be broadcastable")