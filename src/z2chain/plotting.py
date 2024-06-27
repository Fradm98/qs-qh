from qiskit.primitives.base.base_primitive_job import BasePrimitiveJob
from qiskit_ibm_runtime.runtime_job_v2 import RuntimeJobFailureError
import matplotlib.pyplot as plt
import numpy as np

def convert_jobs_to_site_gauge_matrix(jobs_arr):
    site_gauge_observable_matrix = np.zeros((len(jobs_arr), len(first_job_result := jobs_arr[0].result()[0].data.evs)))
    site_gauge_observable_matrix[0] = (1 - first_job_result[::-1])/2
    for i, job in enumerate(jobs_arr[1::], start=1):
        try:
            site_gauge_observable_matrix[i] = (1 - job.result()[0].data.evs[::-1])/2
        except RuntimeJobFailureError as e:
            print(f"WARNING: Jobs with i > {i} failed, the plot will be shorter than expected")
            break
    return site_gauge_observable_matrix[:i]

def save_site_gauge_observable_matrix(site_gauge_observable_matrix, filepath, header=""):
    if isinstance(site_gauge_observable_matrix[0], BasePrimitiveJob):
        site_gauge_observable_matrix = convert_jobs_to_site_gauge_matrix(site_gauge_observable_matrix)

    np.savetxt(filepath, site_gauge_observable_matrix, header=header)

def load_site_gauge_observable_matrix(filepath):
    return np.loadtxt(filepath)

def x_t_plot(site_gauge_observable_matrix, filepath=""):
    if isinstance(site_gauge_observable_matrix[0], BasePrimitiveJob):
        site_gauge_observable_matrix = convert_jobs_to_site_gauge_matrix(site_gauge_observable_matrix)

    plt.rc("text", usetex=True)
    plt.rc("font", size=24, family="serif", weight="bold")

    fig, ax = plt.subplots(1, 1, figsize=[12, 8])

    aspect = site_gauge_observable_matrix.shape[0]/site_gauge_observable_matrix.shape[1]/15

    plt.imshow(site_gauge_observable_matrix, cmap="inferno", aspect=aspect if aspect > 1/2 else 1/2)
    # plt.title(r"Particle & Gauge occupation")
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r"$(1 - \langle Z \rangle)/2$", labelpad=10)
    plt.xlabel(r"Sites")
    plt.ylabel("step")
    plt.tight_layout()

    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")

def discrepancies_plot(exact_site_gauge_observable_matrix, approximated_site_gauge_observable_matrix, filepath=""):
    if isinstance(exact_site_gauge_observable_matrix[0], BasePrimitiveJob):
        exact_site_gauge_observable_matrix = convert_jobs_to_site_gauge_matrix(exact_site_gauge_observable_matrix)
    if isinstance(approximated_site_gauge_observable_matrix[0], BasePrimitiveJob):
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