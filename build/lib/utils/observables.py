from utils.postselection_multithreaded import measure_diagonal_observables
from z2chain.operators import pauli_z_tensor
from multiprocessing import Pool, cpu_count
from itertools import product, combinations
import utils.odr as odr
import numpy as np

def load_single_occupation_correlators_paulis(samples_dict, occupied_qubits, cal_samples_dict=None, confidence=None, shots=None):
    # Generate the pauli observable list involved in the correlator
    try:
        nqubits_sim = len(list(samples_dict.keys())[0])
        nqubits_cal = len(list(cal_samples_dict.keys())[0]) if cal_samples_dict is not None else None
        if nqubits_sim != nqubits_cal:
            raise ValueError("The number of qubits in the samples_dict and cal_samples_dict must be the same")
    except IndexError:
        ncombinations = 2**len(occupied_qubits) - 1
        nan_evals = np.full(ncombinations, np.nan)
        nan_confidence = np.full((ncombinations, 2), np.nan)
        if confidence is not None:
            if cal_samples_dict is not None:
                return (nan_evals, nan_evals), (nan_confidence, nan_confidence), nan_evals
            else:
                return nan_evals, nan_confidence, nan_evals
        else:
            if cal_samples_dict is not None:
                return (nan_evals, nan_evals), nan_evals
            else:
                return nan_evals, nan_evals

    pauli_zs = []
    signs = []
    for i in range(1, len(occupied_qubits)+1):
        these_pauli_zs = []
        for comb in combinations(occupied_qubits, i):
            these_pauli_zs.append(pauli_z_tensor(nqubits_sim, comb))
        signs += [(-1)**i]*len(these_pauli_zs)
        pauli_zs += these_pauli_zs
    if confidence is not None:
        evals, evals_confidence = measure_diagonal_observables(samples_dict, pauli_zs, confidence=confidence)
        if cal_samples_dict is not None:
            cal_evals, cal_evals_confidence = measure_diagonal_observables(cal_samples_dict, pauli_zs, confidence=confidence)
            return (evals, cal_evals), (evals_confidence, cal_evals_confidence), signs
        else:
            return evals, evals_confidence, signs
    else:
        evals = measure_diagonal_observables(samples_dict, pauli_zs)
        if cal_samples_dict is not None:
            cal_evals = measure_diagonal_observables(cal_samples_dict, pauli_zs)
            return (evals, cal_evals), signs
        else:
            return evals, signs
    
def _paulis_results_to_occ_corrs(results, occupied_qubits):
    confidence_present = (len(results[0]) == 3) and isinstance(results[0][0], tuple)
    cal_present = isinstance(results[0][0], tuple)

    occ_qb_paulisz_evals = []
    occ_qb_paulisz_signs = []
    if confidence_present:
        occ_qb_paulisz_confidence = []
        if cal_present:
            cal_occ_qb_paulisz_evals = []
            cal_occ_qb_paulisz_confidence = []
            for result in results:
                occ_qb_paulisz_evals.append(result[0][0])
                cal_occ_qb_paulisz_evals.append(result[0][1])
                occ_qb_paulisz_confidence.append(result[1][0])
                cal_occ_qb_paulisz_confidence.append(result[1][1])
                occ_qb_paulisz_signs.append(result[-1])
        else:
            for result in results:
                occ_qb_paulisz_evals.append(result[0])
                occ_qb_paulisz_confidence.append(result[1])
                occ_qb_paulisz_signs.append(result[-1])
    else:
        if cal_present:
            cal_occ_qb_paulisz_evals = []
            for result in results:
                occ_qb_paulisz_evals.append(result[0][0])
                cal_occ_qb_paulisz_evals.append(result[0][1])
                occ_qb_paulisz_signs.append(result[-1])
        else:
            for result in results:
                occ_qb_paulisz_evals.append(result[0])
                occ_qb_paulisz_signs.append(result[-1])

    nsamples_dicts = len(occ_qb_paulisz_evals)//len(occupied_qubits)
    occupation_correlators_evals = np.zeros((nsamples_dicts, len(occupied_qubits)))
    if cal_present:
        cal_occupation_correlators_evals = np.zeros((nsamples_dicts, len(occupied_qubits)))
        corrected_occupation_correlators_evals = np.zeros((nsamples_dicts, len(occupied_qubits)))
    if confidence_present:
        occupation_correlators_confidence = np.zeros((nsamples_dicts, len(occupied_qubits), 2))
        if cal_present:
            cal_occupation_correlators_confidence = np.zeros((nsamples_dicts, len(occupied_qubits), 2))
            corrected_occupation_correlators_confidence = np.zeros((nsamples_dicts, len(occupied_qubits), 2))
    for i, occ_qb in enumerate(occupied_qubits):
        this_occ_qb_paulisz_evals = np.array(occ_qb_paulisz_evals[i*nsamples_dicts:(i+1)*nsamples_dicts])
        this_occ_qb_paulisz_signs = np.array(occ_qb_paulisz_signs[i*nsamples_dicts:(i+1)*nsamples_dicts])
        occupation_correlators_evals[:, i] = (1 + np.sum(this_occ_qb_paulisz_evals * this_occ_qb_paulisz_signs, axis=1))/2**len(occ_qb)
        if cal_present:
            this_cal_occ_qb_paulisz_evals = np.array(cal_occ_qb_paulisz_evals[i*nsamples_dicts:(i+1)*nsamples_dicts])
            cal_occupation_correlators_evals[:, i] = (1 + np.sum(this_cal_occ_qb_paulisz_evals * this_occ_qb_paulisz_signs, axis=1))/2**len(occ_qb)
            corrected_pauliz_evals = np.zeros_like(this_occ_qb_paulisz_evals)
            if confidence_present:
                this_occ_qb_paulisz_confidence = np.array(occ_qb_paulisz_confidence[i*nsamples_dicts:(i+1)*nsamples_dicts])
                corrected_pauliz_confidence = np.zeros_like(this_occ_qb_paulisz_confidence)
            for j in range(this_cal_occ_qb_paulisz_evals.shape[1]):
                p_arr = odr.compute_depolarizing_probability(this_cal_occ_qb_paulisz_evals[:, j], 1, 0, 0)
                if confidence_present:
                    corrected_pauliz_evals[:, j], corrected_pauliz_confidence[:, j, :] = odr.renormalize_observable(this_occ_qb_paulisz_evals[:, j], p_arr, 0, 0, observable_range=[-1, 1], expected_observable_arr=1, observable_confidence=this_occ_qb_paulisz_confidence[:, j, :])
                else:
                    corrected_pauliz_evals[:, j] = odr.renormalize_observable(this_occ_qb_paulisz_evals[:, j], p_arr, 0, 0, observable_range=[-1, 1], expected_observable_arr=1)
            corrected_occupation_correlators_evals[:, i] = (1 + np.nansum(corrected_pauliz_evals * this_occ_qb_paulisz_signs, axis=1))/2**len(occ_qb)
        if confidence_present:
            confidence_deltas = np.abs(this_occ_qb_paulisz_confidence - this_occ_qb_paulisz_evals.reshape((*this_occ_qb_paulisz_confidence.shape[:2], 1)))
            confidence_deltas = np.abs(np.sum(confidence_deltas * this_occ_qb_paulisz_signs.reshape((*this_occ_qb_paulisz_confidence.shape[:2], 1)), axis=1)/2**len(occ_qb))
            occupation_correlators_confidence[:, i, 0] = occupation_correlators_evals[:, i] - confidence_deltas[:, 0]
            occupation_correlators_confidence[:, i, 1] = occupation_correlators_evals[:, i] + confidence_deltas[:, 1]
            if cal_present:
                this_cal_occ_qb_paulisz_confidence = np.array(cal_occ_qb_paulisz_confidence[i*nsamples_dicts:(i+1)*nsamples_dicts])
                cal_confidence_deltas = np.abs(this_cal_occ_qb_paulisz_confidence - this_cal_occ_qb_paulisz_evals.reshape((*this_cal_occ_qb_paulisz_confidence.shape[:2], 1)))
                cal_confidence_deltas = np.abs(np.sum(cal_confidence_deltas * this_occ_qb_paulisz_signs.reshape((*this_occ_qb_paulisz_confidence.shape[:2], 1)), axis=1)/2**len(occ_qb))
                cal_occupation_correlators_confidence[:, i, 0] = cal_occupation_correlators_evals[:, i] - cal_confidence_deltas[:, 0]
                cal_occupation_correlators_confidence[:, i, 1] = cal_occupation_correlators_evals[:, i] + cal_confidence_deltas[:, 1]
                corrected_confidence_deltas = np.abs(corrected_pauliz_confidence - corrected_pauliz_evals.reshape((*corrected_pauliz_confidence.shape[:2], 1)))
                corrected_confidence_deltas = np.abs(np.nansum(corrected_confidence_deltas * this_occ_qb_paulisz_signs.reshape((*corrected_pauliz_confidence.shape[:2], 1)), axis=1)/2**len(occ_qb))
                corrected_occupation_correlators_confidence[:, i, 0] = corrected_occupation_correlators_evals[:, i] - corrected_confidence_deltas[:, 0]
                corrected_occupation_correlators_confidence[:, i, 1] = corrected_occupation_correlators_evals[:, i] + corrected_confidence_deltas[:, 1]
    
    if cal_present:
        to_return = [corrected_occupation_correlators_evals, occupation_correlators_evals, cal_occupation_correlators_evals]
    else:
        to_return = [occupation_correlators_evals]
    if confidence_present:
        if cal_present:
            to_return = (to_return, [corrected_occupation_correlators_confidence, occupation_correlators_confidence, cal_occupation_correlators_confidence])
        else:
            to_return.append(occupation_correlators_confidence)
    return to_return

def load_occupation_correlators_bypaulis(samples_dicts, occupied_qubits, cal_samples_dicts=None, confidence=None, shots=None):
    if isinstance(samples_dicts, dict):
        samples_dicts = [samples_dicts]
    elif not isinstance(samples_dicts, (list, tuple, np.ndarray)):
        raise ValueError("samples_dicts must be a list, tuple, or numpy array")
    samples_dicts = np.array(samples_dicts, dtype=object)
    if isinstance(occupied_qubits[0], int):
        occupied_qubits = [occupied_qubits]
    elif not isinstance(occupied_qubits, (list, tuple, np.ndarray)):
        raise ValueError("occupied_qubits must be list of list of indices")
    if len(samples_dicts.shape) != 1:
        raise ValueError("samples_dicts must be a 1D array of dictionaries")
    if cal_samples_dicts is not None:
        cal_samples_dicts = np.array(cal_samples_dicts, dtype=object)
        if cal_samples_dicts.shape != samples_dicts.shape:
            raise ValueError("cal_samples_dicts must have the same shape as samples_dicts")

    # results = []
    # for i, occ_qbs in enumerate(occupied_qubits):
    #     for j, samples_dict in enumerate(samples_dicts):
    #         results.append(load_single_occupation_correlators_paulis(samples_dict, occ_qbs, cal_samples_dict=cal_samples_dicts[j] if cal_samples_dicts is not None else None, confidence=confidence, shots=shots))

    pargs = [(samples_dict, occ_qbs, cal_samples_dicts[j] if cal_samples_dicts is not None else None, confidence, shots) for i, occ_qbs in enumerate(occupied_qubits) for j, samples_dict in enumerate(samples_dicts)]
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(load_single_occupation_correlators_paulis, pargs)

    return _paulis_results_to_occ_corrs(results, occupied_qubits)