import numpy as np

def compute_depolarizing_probability(measured_observable_arr, expected_observable_arr, observable_trace, nqubits):
    # PRE 104, 035309 (2021)
    try:
        return (np.array(measured_observable_arr) - np.array(expected_observable_arr))/(observable_trace/2**nqubits - np.array(expected_observable_arr))
    except ValueError:
        raise ValueError("measured_observable_arr expected_observable_arr and observable_trace must be broadcastable")
    
# def compute_depolarizing_probability_std(measured_observable_std_arr, expected_observable_arr, observable_trace, nqubits):
#     try:
#         return np.abs(measured_observable_std_arr/(observable_trace/2**nqubits - np.array(expected_observable_arr)))
#     except ValueError:
#         raise ValueError("measured_observable_arr expected_observable_arr and observable_trace must be broadcastable")
    
def renormalize_observable(measured_observable_arr, p_arr, observable_trace, nqubits, observable_range=None, observable_confidence=None, expected_observable_arr=None, nantol=0.2):
    try:
        one_mask = np.isclose(p_arr, 1)
        p_arr[one_mask] = np.nan
        candidate = ((np.array(measured_observable_arr) - np.array(p_arr)*observable_trace/2**nqubits) / (1 - np.array(p_arr))).astype(float)
        if observable_confidence is not None:
            if expected_observable_arr is None:
                raise ValueError("expected_observable_arr must be provided if observable_confidence is provided")
            new_confidence = np.zeros_like(observable_confidence)
            try:
                left_extremum_perr = (observable_confidence[:, 0] - measured_observable_arr) / (observable_trace/2**nqubits - expected_observable_arr)
                right_extremum_perr = (measured_observable_arr - observable_confidence[:, 1]) / (observable_trace/2**nqubits - expected_observable_arr)
                left_obs_deviation = np.abs((measured_observable_arr - p_arr*observable_trace/2**nqubits)/(1 - p_arr)**2 - observable_trace/2**nqubits/(1 - p_arr))*left_extremum_perr
                right_obs_deviation = np.abs((measured_observable_arr - p_arr*observable_trace/2**nqubits)/(1 - p_arr)**2 - observable_trace/2**nqubits/(1 - p_arr))*right_extremum_perr
            except ValueError:
                raise ValueError("measured_observable_arr expected_observable_arr must be broadcastable")
            new_confidence[:, 0] = candidate - left_obs_deviation
            new_confidence[:, 1] = candidate + right_obs_deviation
        if observable_range is not None:
            mask = np.logical_or(candidate < (1+nantol)*observable_range[0] - nantol, candidate > (1+nantol)*observable_range[1] + nantol)
            candidate[mask] = np.nan
            if observable_confidence is not None:
                new_confidence[mask, :] = np.nan
                return candidate, new_confidence
            else:
                return candidate
        else:
            if observable_confidence is not None:
                return candidate, new_confidence
            else:
                return candidate
    except ValueError:
        raise ValueError("measured_observable_arr, expected_observable_arr and observable_trace must be broadcastable")