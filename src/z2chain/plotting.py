import matplotlib.pyplot as plt
import numpy as np

def x_t_plot(site_guge_observable_matrix):
    plt.rc("text", usetex=True)
    plt.rc("font", size=24, family="serif", weight="bold")

    fig, ax = plt.subplots(1, 1, figsize=[12, 8])

    plt.imshow(site_guge_observable_matrix, cmap="inferno", aspect=site_guge_observable_matrix.shape[0]/site_guge_observable_matrix.shape[1]/15)
    plt.xlabel(r"Sites")
    plt.ylabel("t")
    plt.tight_layout()

def x_t_plot_jobs(jobs_arr):
    site_gauge_observable_matrix = np.zeros((len(jobs_arr), len(jobs_arr[0].result()[0].data.evs)))
    for i, job in enumerate(jobs_arr):
        site_gauge_observable_matrix[i] = (1 - job.result()[0].data.evs)/2
    x_t_plot(site_gauge_observable_matrix)

def discrepancies_plot(exact_site_gauge_observable_matrix, approximated_site_gauge_observable_matrix):
    plt.rc("text", usetex=True)
    plt.rc("font", size=24, family="serif", weight="bold")

    fig, ax = plt.subplots(1, 1, figsize=[12, 8])

    difference = np.abs(exact_site_gauge_observable_matrix - approximated_site_gauge_observable_matrix)

    plt.imshow(difference, cmap="bwr", aspect=exact_site_gauge_observable_matrix.shape[0]/exact_site_gauge_observable_matrix.shape[1]/15, norm="log")
    plt.colorbar()
    plt.xlabel(r"Sites")
    plt.ylabel("t")
    plt.tight_layout()

def discrepancies_plot_jobs(exact_site_gauge_observable_matrix, jobs_arr):
    approximated_site_gauge_observable_matrix = np.zeros((len(jobs_arr), len(jobs_arr[0].result()[0].data.evs)))
    for i, job in enumerate(jobs_arr):
        approximated_site_gauge_observable_matrix[i] = (1 - job.result()[0].data.evs)/2
    discrepancies_plot(exact_site_gauge_observable_matrix, approximated_site_gauge_observable_matrix)