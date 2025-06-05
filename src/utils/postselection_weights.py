from pymatching import Matching
import math
import numpy as np
def decode_with_optimized_weights(string_syndromes, postselection_mask, *, log_errors=False, return_errors=False):
    """
    Decode syndrome using MWPM, with edges adjusted from syndrome bit flip probabilities

    Parameters
    ----------
    string_syndromes : np.ndarray
        A 2D numpy array of shots to decode, with shape=(num_shots, syndrome_length)
    
    postselection_mask : np.ndarray
        Binary parity check matrix, whith shape=(num_syndrome,num_qubits)

    Returns
    -------
    predictions: np.ndarray
        The batch of predictions output by the decoder, a binary numpy array of `dtype=np.uint8` and with shape
        `predictions.shape=(num_shots, self.num_fault_ids)`. `predictions[i, j]=1` iff the decoder predicts that
        fault id `j` was flipped in the shot `i`.

    reconstruction_error: bool
        Whether an error was found due to bad statistics
    """

    w1_ops = dict()
    w2_ops = dict()
    for i in range(postselection_mask.shape[1]): # iterate over physical qubits
        affected_ops = tuple(np.nonzero(postselection_mask[:, i])[0])
        if len(affected_ops) == 2:
            if affected_ops in w2_ops:
                raise ValueError("Two errors with same syndrome, decoding is not possible")
            w2_ops[affected_ops] = i
        elif len(affected_ops) == 1:
            if affected_ops[0] in w1_ops:
                raise ValueError("Two errors with same syndrome, decoding is not possible")
            w1_ops[affected_ops[0]] = i
        else:
            raise ValueError("More than two operators affected by the same error, cannot use MWPM")

    # 'Train' the decoder using statistics of triggered detectors from the experimental data
    # Probability that a detector is triggered
    sample_probs = np.mean(string_syndromes, axis=0)
    # Probability that two detectors are simultaneously triggered
    # Only consider detector pairs that appeared in the original noise model
    sample_corr_probs = dict()
    for (op0, op1) in w2_ops:
        sample_corr_probs[op0, op1] = np.mean(string_syndromes[:, op0] & string_syndromes[:, op1])

    failed_ops = set()
    err_rates = dict()
    # Build new weight-2 errors from detector correlations
    for ops, qubit in w2_ops.items():
        (op0,op1) = ops
        vivj = sample_corr_probs[ops]
        vi = sample_probs[op0]
        vj = sample_probs[op1]
        radicand = 0.25-(vivj-vi*vj)/(1-2*(vi+vj)+4*vivj)
        if radicand < 0 or radicand > 0.25:
            failed_ops.add(ops)
            if log_errors:
                print(f'Warning: cannot determine error rate for bulk qubit {qubit}')
        else:
            pij = 0.5-math.sqrt(radicand)
            err_rates[ops] = pij
    # Build new boundary errors
    for op, qubit in w1_ops.items():
        vi = sample_probs[op]
        prod = 1
        for ops, p in err_rates.items():
            if len(ops) == 2 and op in ops:
                prod *= (1-2*p)
        pi = 0.5+(vi-0.5)/prod
        if pi < 0 or pi > 0.5:
            failed_ops.add((op,))
            if log_errors:
                print(f'Warning: cannot determine error rate for boundary qubit {qubit}')
        else:
            err_rates[(op,)] = pi

    pmean = sum(err_rates.values())/len(err_rates)
    for ops in failed_ops:
        err_rates[ops] = pmean

    decoder = Matching()
    for (op0,op1),qubit in w2_ops.items():
        p = err_rates[(op0,op1)]
        if p > 0:
            decoder.add_edge(op0, op1, qubit, weight=math.log((1-p)/p))
    for op,qubit in w1_ops.items():
        p = err_rates[(op,)]
        if p > 0:
            decoder.add_boundary_edge(op, qubit, weight=math.log((1-p)/p))

    predicted_flips = decoder.decode_batch(string_syndromes)
    if return_errors:
        return predicted_flips, len(failed_ops) > 0
    else:
        return predicted_flips