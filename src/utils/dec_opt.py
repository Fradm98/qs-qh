# ------------------------------------------------- #
#        Distribution Error Correction (DEC)        #
#         (More optimized implementation)           #
#         https://arxiv.org/abs/2501.01953          #
# ------------------------------------------------- #

from multiprocessing import Pool, cpu_count
from qiskit.result import QuasiDistribution
from sympy import fwht, ifwht
import numpy as np

def first_col_update_label(col_label, key):
    num_qubits=len(col_label)
    return bin(int(col_label, 2) ^ int(key, 2))[2:].zfill(num_qubits)

def convert_to_first_col(counts: dict, col_label: str):
    '''The label is the associated column of tilde A. Counts is the associated column values.
    We should convert these to the first column by the symmetry rule described in the paper.'''
    counts={first_col_update_label(col_label, key):value for (key, value) in counts.items()}
    return counts

def filter_counts(counts: dict, num_keep: int) -> dict[str, int]:
    if len(counts) > num_keep:
        counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:num_keep])
    return counts

def fill_label_ints(present_keys_ints: list[int], fill_upto: int) -> np.ndarray:
    present_keys_ints = np.sort(present_keys_ints)
    if fill_upto < len(present_keys_ints):
        raise ValueError("fill_upto must be greater than or equal to the number of present basis states.")
    elif fill_upto >= present_keys_ints[-1]:
        return np.arange(fill_upto)
    else:
        to_add = fill_upto - len(present_keys_ints)
        ints_between_keys = np.diff(present_keys_ints) - 1
        ints_between_keys_cumsum = np.cumsum(ints_between_keys)
        if ints_between_keys_cumsum[-1] > to_add:
            max_key_to_fill_between_index = np.argwhere(ints_between_keys_cumsum >= to_add)[0][0] + 1
            max_key_to_fill_between = present_keys_ints[max_key_to_fill_between_index]
            all_to_fill_between = np.arange(present_keys_ints[0]+1, max_key_to_fill_between)
            present_elements = present_keys_ints[:max_key_to_fill_between_index]
            right_elements = present_keys_ints[max_key_to_fill_between_index:]
        else:
            all_to_fill_between = np.arange(present_keys_ints[0]+1, present_keys_ints[-1])
            present_elements = present_keys_ints
            right_elements = np.array([], dtype=int)
        to_fill_between = np.setdiff1d(all_to_fill_between, present_elements)[:to_add]
        between_elements = np.union1d(present_elements, to_fill_between)
        nfilled_between = len(to_fill_between)
        if nfilled_between < to_add:
            to_fill_before = np.arange(to_add - nfilled_between)
        else:
            to_fill_before = np.array([], dtype=int)
        return np.concatenate((to_fill_before, between_elements, right_elements))  

def samples_to_vecs(samples1: dict, samples2: dict, num_keep: int = None):
    try:
        nqb1 = len(next(iter(samples1.keys())))
    except StopIteration:
        nqb1 = 0
    try:
        nqb2 = len(next(iter(samples2.keys())))
    except StopIteration:
        nqb2 = 0
    if (nqb1 != nqb2) and (nqb1 != 0) and (nqb2 != 0):
        raise ValueError("The number of qubits in samples1 and samples2 must be the same.")
    nqb = max(nqb1, nqb2)
    if num_keep is not None:
        if not float(np.log2(num_keep)).is_integer():
            raise ValueError("num_keep must be a power of 2.")
        if num_keep < max(len(samples1), len(samples2)):
            raise ValueError("num_keep must be greater than or equal to the maximum number of present basis states in samples.")
        if num_keep > 2**nqb:
            raise ValueError(f"num_keep must be less than or equal to 2^nqb (nqb = {nqb}).")
    present_keys = set(samples1.keys()).union(set(samples2.keys()))
    present_keys_ints = np.sort([int(key, 2) for key in present_keys])
    num_keep_keys = 2**int(np.ceil(np.log2(len(present_keys_ints))))
    if num_keep is None:
        num_keep = num_keep_keys
    elif num_keep < num_keep_keys:
        raise NotImplementedError("Proper truncation not implemented YET")
    filled_keys_ints = fill_label_ints(present_keys_ints, num_keep)
    vec1 = np.zeros(num_keep, dtype=int)
    vec2 = np.zeros(num_keep, dtype=int)
    for i, key in enumerate(filled_keys_ints):
        vec1[i] = samples1.get(bin(key)[2:].zfill(nqb), 0)
        vec2[i] = samples2.get(bin(key)[2:].zfill(nqb), 0)
    return vec1, vec2, filled_keys_ints

def hadamard_transform(data, inverse=False):
    # In order to maintain the code vectorized a second dimension must be added.
    original_shape = data.shape
    if not 1 <= len(original_shape) <= 2:
        raise ValueError("The input data must be a one/two-dimensional array.")
    if len(original_shape) == 1:
        x = data[None]
    else:
        x = data
    batch_dim, d = x.shape

    batch_size2 = 2
    while batch_size2 <= d:
        batch_size = batch_size2 // 2
        x = x.reshape(batch_dim, d // batch_size2, batch_size2)
        half1, half2 = x[:, :, :batch_size], x[:, :, batch_size:]
        x = np.concatenate((half1 + half2, half1 - half2), axis=-1)
        batch_size2 = batch_size2 * 2
    
    if inverse:
        x = x / d
    
    return x.reshape(original_shape)

def get_quasi_corrected(noisy_cal_counts_vec, counts_vec, nums_keep=None):
    '''Solves the circulant structure system with FWHT.'''
    if noisy_cal_counts_vec.shape != counts_vec.shape:
        raise ValueError("The shapes of the two vectors must be the same.")
    if nums_keep is not None:
        if len(noisy_cal_counts_vec.shape) < 2:
            raise ValueError("nums_keep can only be provided with more than one counts vector.")
        if len(nums_keep) != noisy_cal_counts_vec.shape[0]:
            raise ValueError("The number of keep values must be the same as the number of experiments.")
        unique_keeps, unique_indices = np.unique(nums_keep, return_index=True)
        quasi_corrected_counts = np.zeros(counts_vec.shape, dtype=float)
        for i in range(len(unique_keeps)-1):
            this_slice = slice(unique_indices[i], unique_indices[i+1])
            fft_calmat_col = hadamard_transform(noisy_cal_counts_vec[this_slice, :unique_keeps[i]])
            fft_noisy_counts = hadamard_transform(counts_vec[this_slice, :unique_keeps[i]])
            quasi_corrected_counts[this_slice, :unique_keeps[i]] = hadamard_transform(np.divide(fft_noisy_counts, fft_calmat_col, out=np.zeros(fft_noisy_counts.shape), where=(fft_calmat_col!=0)), inverse=True)
    else:
        fft_calmat_col = hadamard_transform(noisy_cal_counts_vec)
        fft_noisy_counts = hadamard_transform(counts_vec)
        quasi_corrected_counts=hadamard_transform(np.divide(fft_noisy_counts, fft_calmat_col, out=np.zeros(fft_noisy_counts.shape), where=(fft_calmat_col!=0)), inverse=True)
    return np.array(quasi_corrected_counts, dtype=float)[0:len(noisy_cal_counts_vec)]

def normalize_dist_dict(dist_dict: dict):
    norm = sum(dist_dict.values())
    if norm == 0:
        return {}
    return {key: value / norm for key, value in dist_dict.items()}

def mitigate(payload_noisy_samples, calibration_noisy_samples, calibration_ideal_samples, num_keep=2**20):
    if isinstance(payload_noisy_samples, dict):
        payload_noisy_samples = np.array([payload_noisy_samples], dtype=object)
    elif isinstance(payload_noisy_samples, (tuple, list, np.ndarray)):
        payload_noisy_samples = np.array(payload_noisy_samples, dtype=object)
    else:
        raise TypeError("payload_noisy_samples must be a dict, or iterable of dicts.")
    if isinstance(calibration_noisy_samples, dict):
        calibration_noisy_samples = np.array([calibration_noisy_samples], dtype=object)
    elif isinstance(calibration_noisy_samples, (tuple, list, np.ndarray)):
        calibration_noisy_samples = np.array(calibration_noisy_samples, dtype=object)
    else:
        raise TypeError("calibration_noisy_samples must be a dict, or iterable of dicts.")
    if isinstance(calibration_ideal_samples, dict):
        calibration_ideal_samples = np.array([calibration_ideal_samples], dtype=object)
    elif isinstance(calibration_ideal_samples, (tuple, list, np.ndarray)):
        calibration_ideal_samples = np.array(calibration_ideal_samples, dtype=object)
    else:
        raise TypeError("calibration_ideal_samples must be a dict, or iterable of dicts.")
    if not len(payload_noisy_samples) == len(calibration_noisy_samples) == len(calibration_ideal_samples):
        raise ValueError("The number of samples dicts in payload, calibration and ideal sets must be the same.")
    
    # Extract the calibration column labels and count the number of qubits
    nexperiments = len(payload_noisy_samples)
    calibration_column_labels = [list(isamples.keys())[0] for isamples in calibration_ideal_samples]
    nqubits = [len(label) for label in calibration_column_labels]

    # Compute the max length of the samples dicts
    max_basis_dim = 0
    for psamples, csamples in zip(payload_noisy_samples, calibration_noisy_samples):
        this_max_nbasis = max(len(psamples), len(csamples))
        if this_max_nbasis > max_basis_dim:
            max_basis_dim = this_max_nbasis
    max_basis_dim = 2**(int(np.ceil(np.log2(max_basis_dim))) + 1)

    # Generate the distribution vectors with correct ordering and length
    pvecs = np.zeros((nexperiments, num_keep), dtype=int)
    cvecs = np.zeros((nexperiments, num_keep), dtype=int)
    int_labels = np.zeros((nexperiments, num_keep), dtype=int)
    nums_keep = np.zeros(nexperiments, dtype=int)
    for i, (psamples, csamples) in enumerate(zip(payload_noisy_samples, calibration_noisy_samples)):
        psamples = convert_to_first_col(psamples, calibration_column_labels[i])
        psamples = filter_counts(psamples, num_keep)
        csamples = filter_counts(csamples, num_keep)
        this_pvec, this_cvec, this_int_labels = samples_to_vecs(psamples, csamples, num_keep)
        this_num_keep = len(this_pvec)
        pvecs[i, :this_num_keep] = this_pvec
        cvecs[i, :this_num_keep] = this_cvec
        int_labels[i, :this_num_keep] = this_int_labels
        nums_keep[i] = this_num_keep
    
    # Get the corrected payload distribution
    quasi_corrected_pvecs = get_quasi_corrected(cvecs, pvecs)
    corrected_quasi_dist_dicts = [{bin(label)[2:].zfill(nqubits[i]):prob for label, prob in zip(int_labels[i], quasi_corrected_pvecs[i][:nums_keep[i]])} for i in range(nexperiments)]
    corrected_dist_dicts = [QuasiDistribution(quasi_dist_dict).nearest_probability_distribution().binary_probabilities(nqubits[i]) for i, quasi_dist_dict in enumerate(corrected_quasi_dist_dicts)]
    corrected_dist_dicts = [normalize_dist_dict(corrected_dist_dict) for corrected_dist_dict in corrected_dist_dicts]
    return np.array(corrected_dist_dicts, dtype=object) if len(corrected_dist_dicts) > 1 else corrected_dist_dicts[0]