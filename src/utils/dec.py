# ------------------------------------------------- #
#        Distribution Error Correction (DEC)        #
#         https://arxiv.org/abs/2501.01953          #
# ------------------------------------------------- #

from utils.misc import starmap_with_progress
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

def filter_counts(counts, num_keep):
    '''Returns a dict with only the largest num_keep elements.'''
    list_counts=sorted(counts.items(), key=lambda x: -x[1])
    filtered_counts=list_counts[:num_keep:]
    return {k:v for (k,v) in filtered_counts}

def fill_counts_range(counts1: dict, counts2: dict, num_keep, num_qubits):
    '''We embed the counts into a power of 2 number of basis states by padding with zeros.
    Some things to note: This is necessary to match the keys for both dict. Also,
    since FWHT solvers only see a vector, when they pad they can do it in an
    unphysical fashion (e.g., 111 is in the distribution so no padding should occur after the 111.)
    args: num_keep: should be a power of 2. Both list should have the same keys for the
    inverse Fourier transforms to generate correct results.'''
    counts1_keys=set(counts1.keys())
    counts2_keys=set(counts2.keys())
    new_counts1={}
    new_counts2={}
    exist_keys=list(counts1_keys.union(counts2_keys))
    exist_keys=[int(elem, 2) for elem in exist_keys]
    exist_keys=sorted(exist_keys)
    max_key=exist_keys[-1]
    min_key=exist_keys[0]
    current_len=len(exist_keys)
    if num_keep>=max_key:
        max_key=num_keep
        all_keys=list(range(num_keep))
    else: #num_keep<max_key. we will pad between until the size is right
        if num_keep<current_len: #the current len exceeds what was kept.
            # go to the nearest larger power of 2.
            power=int(np.ceil(np.log2(current_len)))
            max_key=num_keep=2**power
        all_possible_between=iter(range(min_key, max_key))
        all_possible_before=iter(range(0, min_key))
        new_keys_between=[]
        new_keys_before=[]
        # we should pad with elements between the min and max keys before outside.
        while len(new_keys_between)<num_keep-current_len:
            try:
                next_elem=next(all_possible_between)
                while next_elem in exist_keys:
                    try:
                        next_elem=next(all_possible_between)
                    except StopIteration:
                        raise StopIteration
            except StopIteration:
                break
            new_keys_between.append(next_elem)
        while len(new_keys_between)+len(new_keys_before)+len(exist_keys)<num_keep:
            # pad with before elements
            next_elem=next(all_possible_before)
            new_keys_before.append(next_elem)

        all_keys=exist_keys+new_keys_between+new_keys_before
        
    all_keys=[bin(elem)[2::].zfill(num_qubits) for elem in all_keys]

    for k in all_keys:
        new_counts1[k]=counts1.get(k,0)
        new_counts2[k]=counts2.get(k,0)
    assert len(all_keys)==len(new_counts1)
    assert len(all_keys)==len(new_counts2)

    return sorted(list(all_keys), key=lambda x: int(x)), new_counts1, new_counts2

def filled_dict_to_vec(filled_samples_dict):
    if not float(np.log2(len(filled_samples_dict))).is_integer():
        raise ValueError("The number of keys in the samples dict must be a power of 2.")
    sorted_distribution = sorted(list(filled_samples_dict.items()), key=lambda x: int(x[0], 2))
    return np.array([el[1] for el in sorted_distribution])

def hadamard_transform(data, inverse=True):
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
        x = x / np.sqrt(d)
    
    return x.reshape(original_shape)

def multiprocess_convert_dicts_to_vecs(payload_noisy_samples, calibration_noisy_samples, first_column_labels, num_keep=2**20):
    if not isinstance(payload_noisy_samples, (list, tuple, np.ndarray)) or not isinstance(calibration_noisy_samples, (list, tuple, np.ndarray)):
        raise ValueError("More than one samples dicts must be provided to use multiprocess.")
    if len(payload_noisy_samples) != len(calibration_noisy_samples) != len(first_column_labels):
        raise ValueError("The payload, calibration samples and first_column_labels lists must have the same length.")
    
    nqubits = [len(label) for label in first_column_labels]
    nums_keep = [2**int(np.ceil(np.log2(min(2**nqubits[i], num_keep, max(len(payload_noisy_samples[i]), len(calibration_noisy_samples[i])))))) for i in range(len(payload_noisy_samples))]
    
    with Pool(processes=cpu_count()) as pool:
        payload_noisy_samples = pool.starmap(convert_to_first_col, zip(payload_noisy_samples, first_column_labels))
        payload_noisy_samples = pool.starmap(filter_counts, zip(payload_noisy_samples, nums_keep))
        calibration_noisy_samples = pool.starmap(filter_counts, zip(calibration_noisy_samples, nums_keep))
        filled_labels_and_dicts = pool.starmap(fill_counts_range, zip(payload_noisy_samples, calibration_noisy_samples, nums_keep, nqubits))
        print(filled_labels_and_dicts[0][1::])
        labels = [el[0] for el in filled_labels_and_dicts]
        filled_payload_samples_vec = np.array(pool.starmap(filled_dict_to_vec, [el[1] for el in filled_labels_and_dicts]))
        filled_calibration_samples_vec = np.array(pool.starmap(filled_dict_to_vec, [el[2] for el in filled_labels_and_dicts]))

    return labels, filled_payload_samples_vec, filled_calibration_samples_vec

def get_quasi_corrected(noisy_noise_est_vec, counts_vec):
    '''Solves the circulant structure system with FWHT.'''
    fft_calmat_col=hadamard_transform(noisy_noise_est_vec)
    fft_noisy_counts=hadamard_transform(counts_vec)
    quasi_corrected_counts=hadamard_transform(np.divide(fft_noisy_counts, fft_calmat_col, out=np.zeros(len(fft_noisy_counts)), where=(fft_calmat_col!=0)), inverse=True)
    return np.array(quasi_corrected_counts, dtype=float)[0:len(noisy_noise_est_vec)]

def normalize_dist_dict(dist_dict: dict):
    norm = sum(dist_dict.values())
    if norm == 0:
        return {}
    return {key: value / norm for key, value in dist_dict.items()}

def mitigate(payload_noisy_samples, calibration_noisy_samples, calibration_ideal_samples, num_keep=2**20):
    if len(calibration_ideal_samples) != 1:
        raise ValueError("Calibration samples must only contain samples from a single basis state.")
    if sum(payload_noisy_samples.values()) != sum(calibration_noisy_samples.values()) != sum(calibration_ideal_samples.values()):
        raise ValueError("All samples dicts must contain the same number of shots.")
    shots = sum(payload_noisy_samples.values())
    calibration_column_label = list(calibration_ideal_samples.keys())[0]
    nqubits = len(calibration_column_label)

    # Generate the distribution vectors with correct ordering and length
    payload_noisy_samples = convert_to_first_col(payload_noisy_samples, calibration_column_label)
    payload_noisy_samples = filter_counts(payload_noisy_samples, num_keep)
    calibration_noisy_samples = filter_counts(calibration_noisy_samples, num_keep)
    final_labels, filled_payload_noisy_samples, filled_calibration_noisy_samples = fill_counts_range(payload_noisy_samples, calibration_noisy_samples, 2**int(np.ceil(np.log2(min(2**nqubits, num_keep, max(len(payload_noisy_samples), len(calibration_noisy_samples)))))), nqubits)
    filled_payload_samples_vec = filled_dict_to_vec(filled_payload_noisy_samples)
    filled_calibration_samples_vec = filled_dict_to_vec(filled_calibration_noisy_samples)

    # Get the corrected payload distribution
    payload_quasi_corrected_dist_vec = get_quasi_corrected(filled_calibration_samples_vec, filled_payload_samples_vec)
    payload_quasi_dist_dict = {label:prob for label, prob in zip(final_labels, payload_quasi_corrected_dist_vec)}
    payload_quasi_dist = QuasiDistribution(payload_quasi_dist_dict)
    payload_dist_dict = payload_quasi_dist.nearest_probability_distribution().binary_probabilities(nqubits)
    payload_dist_dict = normalize_dist_dict(payload_dist_dict)

    return payload_dist_dict

def multiprocess_mitigate(payload_noisy_samples, calibration_noisy_samples, calibration_ideal_samples, num_keep=2**20):
    if not isinstance(payload_noisy_samples, (list, tuple, np.ndarray)):
        raise ValueError("payload_noisy_samples must be a list or tuple of dictionaries.")
    else:
        payload_noisy_samples = np.array(payload_noisy_samples, dtype=object)
    if not isinstance(calibration_noisy_samples, (list, tuple, np.ndarray)):
        raise ValueError("calibration_noisy_samples must be a list or tuple of dictionaries.")
    else:
        calibration_noisy_samples = np.array(calibration_noisy_samples, dtype=object)
    if not isinstance(calibration_ideal_samples, (list, tuple, dict, np.ndarray)):
        raise ValueError("calibration_ideal_samples must be a list, tuple of dicts or a single dict.")
    elif isinstance(calibration_ideal_samples, dict):
        calibration_ideal_samples = np.array([calibration_ideal_samples]*len(payload_noisy_samples), dtype=object)
    else:
        calibration_ideal_samples = np.array(calibration_ideal_samples, dtype=object)

    if len(payload_noisy_samples) != len(calibration_noisy_samples) != len(calibration_ideal_samples):
        raise ValueError("All samples lists must have the same length.")

    with Pool(processes=cpu_count()) as pool:
        # args = zip(payload_noisy_samples, calibration_noisy_samples, calibration_ideal_samples, [num_keep]*len(payload_noisy_samples))
        # mitigated_samples = starmap_with_progress(pool, mitigate, args, len(payload_noisy_samples))
        mitigated_samples = pool.starmap(mitigate, zip(payload_noisy_samples, calibration_noisy_samples, calibration_ideal_samples, [num_keep]*len(payload_noisy_samples)))
    return np.array(mitigated_samples, dtype=object)