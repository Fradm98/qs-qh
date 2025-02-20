from utils.postselection_multithreaded import combine_samples_dicts
from pymatching import Matching
import numpy as np

def bias_corrected_get_recovered_samples_dicts(samples_dict, postselection_ops, basis, flip_threshold=None, flipped_syndromes=None):
    if len(samples_dict) == 0:
        return {}
    strings_arr = []
    string_counts = []
    for string, counts in samples_dict.items():
        strings_arr.append([int(c) for c in string])
        string_counts.append(counts)
    strings_arr = np.array(strings_arr, dtype=int)
    string_counts = np.array(string_counts, dtype=int)
    postselection_mask = np.array([[int(str(opel) == basis) for opel in operator] for operator in postselection_ops])
    if flipped_syndromes is not None:
        if len(flipped_syndromes) != postselection_mask.shape[0]:
            raise ValueError("flipped syndromes must be the same length as postselection_ops")
    decoder = Matching(postselection_mask)
    string_syndromes = (strings_arr @ postselection_mask.T) % 2
    if flipped_syndromes is not None:
        string_syndromes = (string_syndromes + flipped_syndromes) % 2
    predicted_flips = decoder.decode_batch(string_syndromes)

    # Remove strings with too many flips
    if flip_threshold is not None:
        n_flips = np.sum(predicted_flips, axis=1)
        valid_mask = np.less_equal(n_flips, flip_threshold)
        strings_arr = strings_arr[valid_mask, :]
        predicted_flips = predicted_flips[valid_mask, :]
        string_counts = string_counts[valid_mask]
    
    # Convolution to detect isolated flips
    matter_predicted_flips = predicted_flips[:, ::2]
    kernel = np.array([1, 1, 1])
    convolution = np.apply_along_axis(lambda ax: np.convolve(ax, kernel, mode="same"), axis=1, arr=matter_predicted_flips)
    isolated_flips_inds = np.argwhere(np.logical_and(np.equal(matter_predicted_flips, 1), np.equal(convolution, 1)))
    isolated_flips_inds[:, 1] = 2*isolated_flips_inds[:, 1]
    unique_isolated_flips_row_inds, unique_isolated_flips_row_counts = np.unique(isolated_flips_inds[:, 0], return_counts=True)
    not_isolated_mask = np.ones(strings_arr.shape[0], dtype=bool)
    not_isolated_mask[unique_isolated_flips_row_inds] = False

    non_biased_strings = strings_arr[not_isolated_mask, :]
    non_biased_corrections = predicted_flips[not_isolated_mask, :]
    non_biased_strings_counts = string_counts[not_isolated_mask]

    biased_strings = strings_arr[unique_isolated_flips_row_inds, :]
    biased_strings_corrections = predicted_flips[unique_isolated_flips_row_inds, :]
    biased_strings_counts = string_counts[unique_isolated_flips_row_inds]
    nbiased_samples = np.sum(biased_strings_counts)

    isolated_flips_row_range = np.arange(isolated_flips_inds.shape[0])
    isolated_flips_unique_weights = (biased_strings_counts / np.sum(biased_strings_counts)) / unique_isolated_flips_row_counts
    isolated_flips_weights = np.repeat(isolated_flips_unique_weights, unique_isolated_flips_row_counts)
    isolated_flips_row_ind_sample = np.random.choice(isolated_flips_row_range, size=nbiased_samples//4, p=isolated_flips_weights)
    isolated_flips_ind_sample = isolated_flips_inds[isolated_flips_row_ind_sample, :]
    unique_isolated_flips_row_ind_sample, unique_isolated_flips_row_ind_sample_counts = np.unique(isolated_flips_ind_sample[:, 0], return_counts=True)
    to_correct_bias_strings = strings_arr[unique_isolated_flips_row_ind_sample, :]
    to_correct_bias_strings_corrections = predicted_flips[unique_isolated_flips_row_ind_sample, :]
    not_to_correct_bias_strings_counts = string_counts[unique_isolated_flips_row_ind_sample] - unique_isolated_flips_row_ind_sample_counts
    
    # Treat non-biased and not-to-correct strings equally
    # dd: Direct Decoding
    dd_strings = np.vstack((non_biased_strings, to_correct_bias_strings))
    dd_corrections = np.vstack((non_biased_corrections, to_correct_bias_strings_corrections))
    dd_strings_counts = np.hstack((non_biased_strings_counts, not_to_correct_bias_strings_counts))
    dd_recovered_strings = (dd_strings + dd_corrections) % 2
    unbiased_samples_dict = {}
    fmt_str = "%i"*dd_recovered_strings.shape[1]
    for i, string in enumerate(dd_recovered_strings):
        string_str = fmt_str % tuple(string)
        if string_str in unbiased_samples_dict:
            unbiased_samples_dict[string_str] += dd_strings_counts[i]
        else:
            unbiased_samples_dict[string_str] = dd_strings_counts[i]

    # Correct the biased strings
    bias_corrected_samples_dict = {}
    for isolated_flip_indices in isolated_flips_ind_sample:
        this_flip = predicted_flips[isolated_flip_indices[0]].copy()
        this_string = strings_arr[isolated_flip_indices[0]]
        if isolated_flip_indices[1] > 1:
            this_flip[isolated_flip_indices[1]-2:isolated_flip_indices[1]+1] = (this_flip[isolated_flip_indices[1]-2:isolated_flip_indices[1]+1] + 1) % 2
        this_corrected_string = (this_string + this_flip) % 2
        this_corrected_str = fmt_str % tuple(this_corrected_string)
        if this_corrected_str in bias_corrected_samples_dict:
            bias_corrected_samples_dict[this_corrected_str] += 1
        else:
            bias_corrected_samples_dict[this_corrected_str] = 1

    # Combine dictionaries
    recovered_samples_dict = combine_samples_dicts(unbiased_samples_dict, bias_corrected_samples_dict)
    return recovered_samples_dict