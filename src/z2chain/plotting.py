from qiskit.primitives.base.base_primitive_job import BasePrimitiveJob
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit_ibm_runtime.runtime_job import RuntimeJob
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

def convert_jobs_to_site_gauge_matrix(jobs_arr):
    printed_warning_header = False
    first_success_index = 0
    while first_success_index < len(jobs_arr):
        try:
            first_success_job = jobs_arr[first_success_index]
            if not first_success_job.in_final_state() and type(first_success_job) is not PrimitiveJob:
                raise TimeoutError("Some jobs are still to be finished")
            else:
                first_success_result = first_success_job.result()[0].data.evs[::-1]
            site_gauge_observable_matrix = np.full((len(jobs_arr), len(first_success_result)), np.nan)
            break
        except:
            if not printed_warning_header: 
                print("WARNING: Some jobs failed, the plot will be incomplete\nFAILED JOBS\nIndex | ID")
                printed_warning_header = True
            print(f"{str(first_success_index).ljust(6)}| {jobs_arr[first_success_index].job_id()}")
            first_success_index += 1
            continue
    site_gauge_observable_matrix[first_success_index] = (1 - first_success_result[::-1])/2
    for i, job in enumerate(jobs_arr[first_success_index::], start=first_success_index):
        try:
            if not job.in_final_state() and type(first_success_job) is not PrimitiveJob:
                raise TimeoutError("Some jobs are still to be finished")
            else:
                this_job_result = job.result()[0].data.evs[::-1]
            site_gauge_observable_matrix[i] = (1 - this_job_result)/2
        except:
            if not printed_warning_header: 
                print("WARNING: Some jobs failed, the plot will be incomplete\nFAILED JOBS\nIndex | ID")
                printed_warning_header = True
            print(f"{str(i).ljust(6)}| {job.job_id()}")
            continue
    return site_gauge_observable_matrix[:i]

def save_site_gauge_observable_matrix(site_gauge_observable_matrix, filepath, header=""):
    if isinstance(site_gauge_observable_matrix[0], BasePrimitiveJob):
        site_gauge_observable_matrix = convert_jobs_to_site_gauge_matrix(site_gauge_observable_matrix)

    np.savetxt(filepath, site_gauge_observable_matrix, header=header)

def load_site_gauge_observable_matrix(filepath):
    return np.loadtxt(filepath)

def x_t_plot(site_gauge_observable_matrix, nxticks=5, filepath=""):
    if isinstance(site_gauge_observable_matrix[0], BasePrimitiveJob) or isinstance(site_gauge_observable_matrix[0], RuntimeJob):
        site_gauge_observable_matrix = convert_jobs_to_site_gauge_matrix(site_gauge_observable_matrix)

    plt.rc("text", usetex=True)
    plt.rc("font", size=24, family="serif", weight="bold")

    fig, ax = plt.subplots(1, 1, figsize=[12, 8])

    aspect = site_gauge_observable_matrix.shape[0]/site_gauge_observable_matrix.shape[1]/15

    plt.imshow(site_gauge_observable_matrix, cmap="inferno", aspect=aspect if aspect > 1/2 else 1/2, vmax=1, vmin=0)
    plt.title(r"Particle \& Gauge occupation")
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r"$(1 - \langle Z \rangle)/2$", labelpad=10)
    plt.xlabel(r"Sites")
    plt.ylabel("step")
    plt.xticks(np.round(np.linspace(0, site_gauge_observable_matrix.shape[1]-1, nxticks)).astype(int), np.round(np.linspace(1, site_gauge_observable_matrix.shape[1], nxticks)).astype(int))
    plt.tight_layout()

    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")

def discrepancies_plot(exact_site_gauge_observable_matrix, approximated_site_gauge_observable_matrix, filepath=""):
    if isinstance(exact_site_gauge_observable_matrix[0], BasePrimitiveJob) or isinstance(exact_site_gauge_observable_matrix[0], RuntimeJob):
        exact_site_gauge_observable_matrix = convert_jobs_to_site_gauge_matrix(exact_site_gauge_observable_matrix)
    if isinstance(approximated_site_gauge_observable_matrix[0], BasePrimitiveJob) or isinstance(approximated_site_gauge_observable_matrix[0], RuntimeJob):
        approximated_site_gauge_observable_matrix = convert_jobs_to_site_gauge_matrix(approximated_site_gauge_observable_matrix)

    max_len = np.min([exact_site_gauge_observable_matrix.shape[0], approximated_site_gauge_observable_matrix.shape[0]])
    exact_site_gauge_observable_matrix = exact_site_gauge_observable_matrix[:max_len]
    approximated_site_gauge_observable_matrix = approximated_site_gauge_observable_matrix[:max_len]

    plt.rc("text", usetex=True)
    plt.rc("font", size=24, family="serif", weight="bold")

    fig, ax = plt.subplots(1, 1, figsize=[12, 8])

    difference = np.abs(exact_site_gauge_observable_matrix - approximated_site_gauge_observable_matrix)

    aspect = difference.shape[0]/difference.shape[1]/15

    plt.imshow(difference, cmap="bwr", aspect=aspect if aspect > 1/2 else 1/2, norm="log")
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r"$| \langle n \rangle_{\mathrm{exact}} - \langle n \rangle_{\mathrm{Trotter}} |$", labelpad=10)
    plt.xlabel(r"Sites")
    plt.ylabel("step")
    plt.tight_layout()

    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")

def plot_n_discarded_samples(samples_dicts, postselected_samples_dicts, x_arr=None, xlabel=None, regression=False, filepath=""):
    plt.rc("text", usetex=True)
    plt.rc("font", size=22, family="serif", weight="bold")

    if type(samples_dicts) not in [list, tuple]:
        samples_dicts = [samples_dicts]
    if type(postselected_samples_dicts) not in [list, tuple]:
        postselected_samples_dicts = [postselected_samples_dicts]

    if len(samples_dicts) != len(postselected_samples_dicts):
        raise ValueError("Samples and Postselected samples dict are not the same lenght")

    if x_arr is None:
        x_arr = np.arange(1, len(samples_dicts)+1)
    else:
        if len(x_arr) != len(samples_dicts):
            raise ValueError("x_arr mush be the same length as samples_dicts")
    ntotal_samples_arr = np.zeros(len(samples_dicts))
    npostselected_samples_arr = np.zeros(len(samples_dicts))

    for i, (samples_dict, postselected_samples_dict) in enumerate(zip(samples_dicts, postselected_samples_dicts)):
        nqubits_samples = len(list(samples_dict.keys())[0])
        nqubits_postselected_samples = len(list(postselected_samples_dict.keys())[0])
        if nqubits_samples != nqubits_postselected_samples:
            raise ValueError("Sample dicts in the same posititions must have the same number of qubits")
        ntotal_samples_arr[i] = np.sum(list(samples_dict.values()))
        npostselected_samples_arr[i] = np.sum(list(postselected_samples_dict.values()))
    
    ratio_arr = npostselected_samples_arr/ntotal_samples_arr
    fix, ax = plt.subplots(figsize=[8, 5])
    if regression:
        reg_func = lambda x, a, b: b*np.exp(-a*x)
        popt, pcov = sp.optimize.curve_fit(reg_func, x_arr, ratio_arr, p0=[1, 1])
        plt.plot(x_arr, reg_func(x_arr, *popt), "--", linewidth=2, color="darkslateblue", label="$b e^{-ax}$\n"+f"$a={popt[0]:.03f}$\n"+f"$b = {popt[1]:0.2f}$")
    plt.plot(x_arr, ratio_arr, "x", markersize=10, markeredgewidth=2, color="darkorange")
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6)
    plt.ylabel(r"$n_{\mathrm{post}}/n_{\mathrm{total}}$", labelpad=10)
    if regression:
        plt.legend(prop={"size":18})
    if xlabel is not None:
        plt.xlabel(xlabel, labelpad=10)
    else:
        plt.xlabel("Sample sets", labelpad=10)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")