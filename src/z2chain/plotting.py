from qiskit.primitives.base.base_primitive_job import BasePrimitiveJob
from qiskit_ibm_runtime.runtime_job_v2 import RuntimeJobFailureError
from qiskit_ibm_runtime.runtime_job import RuntimeJob
import matplotlib.pyplot as plt
import numpy as np

def convert_jobs_to_site_gauge_matrix(jobs_arr):
    printed_warning_header = False
    first_success_index = 0
    while first_success_index < len(jobs_arr):
        try:
            first_success_job = jobs_arr[first_success_index]
            if not first_success_job.in_final_state():
                raise TimeoutError("Some jobs are still to be finished")
            else:
                first_success_result = first_success_job.result()[0].data.evs[::-1]
            site_gauge_observable_matrix = np.full((len(jobs_arr), len(first_success_result)), np.nan)
            break
        except RuntimeJobFailureError:
            if not printed_warning_header: 
                print("WARNING: Some jobs failed, the plot will be incomplete\nFAILED JOBS\nIndex | ID")
                printed_warning_header = True
            print(f"{str(first_success_index).ljust(6)}| {jobs_arr[first_success_index].job_id()}")
            first_success_index += 1
            continue
    site_gauge_observable_matrix[first_success_index] = (1 - first_success_result[::-1])/2
    for i, job in enumerate(jobs_arr[first_success_index::], start=first_success_index):
        try:
            if not job.in_final_state():
                raise TimeoutError("Some jobs are still to be finished")
            else:
                this_job_result = job.result()[0].data.evs[::-1]
            site_gauge_observable_matrix[i] = (1 - this_job_result)/2
        except RuntimeJobFailureError:
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

    plt.imshow(site_gauge_observable_matrix, cmap="inferno", aspect=aspect if aspect > 1/2 else 1/2)
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