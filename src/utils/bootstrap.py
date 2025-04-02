import numpy as np
import scipy as sp
    
def bootstrap_sample_dict_mean(samples_dict, string_expectation_values, confidence=0.95, n_resamples=1000, return_std=False):
    if confidence < 0 or confidence > 1:
        raise ValueError("Confidence must be between 0 and 1")
    if n_resamples < 1:
        raise ValueError("Number of resamples must be at least 1")
    
    observables_eigenvalues_counts = np.zeros((2, string_expectation_values.shape[1]), dtype=int)
    string_counts = np.array(list(samples_dict.values()))
    total_string_counts = np.sum(string_counts)
    for i in range(string_expectation_values.shape[1]):
        observables_eigenvalues_counts[0, i] = np.sum(string_counts[np.equal(string_expectation_values[:, i], 1)])
        observables_eigenvalues_counts[1, i] = total_string_counts - observables_eigenvalues_counts[0, i]
    
    confidence_intervals = []
    if return_std: stds = []
    for i in range(string_expectation_values.shape[1]):
        # Bootstrap resampling
        bootstrap_set = np.zeros(n_resamples)
        for j in range(n_resamples):
            this_batch_size = n_resamples - j
            bootstrap_set[j] = np.mean(np.random.choice([1, -1], p=[observables_eigenvalues_counts[0, i] / total_string_counts, observables_eigenvalues_counts[1, i] / total_string_counts], size=this_batch_size, replace=True))
        
        # Calculate percentile interval
        alpha = (1 - confidence)/2

        # Correct for bias and acceleration
        original_expectation_value = (observables_eigenvalues_counts[0, i] - observables_eigenvalues_counts[1, i]) / total_string_counts
        percentile = (np.sum(np.less(bootstrap_set, original_expectation_value)) + np.sum(np.less_equal(bootstrap_set, original_expectation_value))) / 2 / n_resamples
        z0_hat = sp.special.ndtri(percentile)

        jackknife_mean = observables_eigenvalues_counts[0, i]/total_string_counts*(((observables_eigenvalues_counts[0, i]-1) - observables_eigenvalues_counts[1, i]) / total_string_counts) + observables_eigenvalues_counts[1, i]/total_string_counts*((observables_eigenvalues_counts[0, i] - (observables_eigenvalues_counts[1, i]-1)) / total_string_counts)
        jackknife_samples = np.array([((observables_eigenvalues_counts[0, i]-1) - observables_eigenvalues_counts[1, i]) / total_string_counts]*observables_eigenvalues_counts[0, i] + [(observables_eigenvalues_counts[0, i] - (observables_eigenvalues_counts[1, i]-1)) / total_string_counts]*observables_eigenvalues_counts[1, i])

        U_sample =  (n_resamples - 1) * (jackknife_mean - jackknife_samples)
        num = np.sum(U_sample**3) / n_resamples**3
        den = np.sum(U_sample**2) / n_resamples**2
        if np.isclose(den, 0):
            if np.isclose(num, 0):
                a_hat = 1/6
            else:
                confidence_intervals.append((np.nan, np.nan))
                if return_std: stds.append(np.nan)
                continue
        else:
            a_hat = num / (6 * den**(3/2))

        z_alpha = sp.special.ndtri(alpha)
        z1_alpha = -z_alpha
        num1 = z0_hat + z_alpha
        alpha1 = sp.special.ndtr(z0_hat + num1 / (1 - a_hat * num1))
        num2 = z0_hat + z1_alpha
        alpha2 = sp.special.ndtr(z0_hat + num2 / (1 - a_hat * num2))

        ci_l = np.percentile(bootstrap_set, 100*alpha1)
        ci_u = np.percentile(bootstrap_set, 100*alpha2)
        ci_l, ci_u = 2*original_expectation_value - ci_u, 2*original_expectation_value - ci_l

        confidence_intervals.append((ci_l, ci_u))
        if return_std: stds.append(np.std(bootstrap_set))

    if return_std:
        return np.array(confidence_intervals), stds
    else:
        return np.array(confidence_intervals)