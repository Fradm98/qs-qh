from qiskit.primitives.primitive_job import PrimitiveJob
import matplotlib.pyplot as plt
import numpy as np

def convert_jobs_to_site_gauge_matrix(jobs_arr):
    site_gauge_observable_matrix = np.zeros((len(jobs_arr), len(jobs_arr[0].result()[0].data.evs)))
    for i, job in enumerate(jobs_arr):
        site_gauge_observable_matrix[i] = (1 - job.result()[0].data.evs)/2
    return site_gauge_observable_matrix

def x_t_plot(site_gauge_observable_matrix):
    if type(site_gauge_observable_matrix[0]) is PrimitiveJob:
        site_gauge_observable_matrix = convert_jobs_to_site_gauge_matrix(site_gauge_observable_matrix)

    plt.rc("text", usetex=True)
    plt.rc("font", size=24, family="serif", weight="bold")

    fig, ax = plt.subplots(1, 1, figsize=[12, 8])

    plt.imshow(site_gauge_observable_matrix, cmap="inferno", aspect=site_gauge_observable_matrix.shape[0]/site_gauge_observable_matrix.shape[1]/15)
    plt.xlabel(r"Sites")
    plt.ylabel("t")
    plt.tight_layout()

def discrepancies_plot(exact_site_gauge_observable_matrix, approximated_site_gauge_observable_matrix):
    if type(exact_site_gauge_observable_matrix[0]) is PrimitiveJob:
        exact_site_gauge_observable_matrix = convert_jobs_to_site_gauge_matrix(exact_site_gauge_observable_matrix)
    if type(approximated_site_gauge_observable_matrix[0]) is PrimitiveJob:
        approximated_site_gauge_observable_matrix = convert_jobs_to_site_gauge_matrix(approximated_site_gauge_observable_matrix)

    plt.rc("text", usetex=True)
    plt.rc("font", size=24, family="serif", weight="bold")

    fig, ax = plt.subplots(1, 1, figsize=[12, 8])

    difference = np.abs(exact_site_gauge_observable_matrix - approximated_site_gauge_observable_matrix)

    plt.imshow(difference, cmap="bwr", aspect=exact_site_gauge_observable_matrix.shape[0]/exact_site_gauge_observable_matrix.shape[1]/15, norm="log")
    plt.colorbar()
    plt.xlabel(r"Sites")
    plt.ylabel("t")
    plt.tight_layout()