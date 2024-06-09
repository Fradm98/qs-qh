import matplotlib.pyplot as plt
import numpy as np

def x_t_plot(site_guge_observable_matrix):
    plt.rc("text", usetex=True)
    plt.rc("font", size=24, family="serif", weight="bold")

    fig, ax = plt.subplots(1, 1, figsize=[8, 8])

    plt.imshow(site_guge_observable_matrix, cmap="inferno", aspect=site_guge_observable_matrix.shape[0]/site_guge_observable_matrix.shape[1]/15)
    plt.xlabel(r"Matter (Even) \& Gauge (Odd) sites")
    plt.ylabel("t")

def x_t_plot_jobs(jobs_arr):
    site_gauge_observable_matrix = np.zeros((len(jobs_arr), len(jobs_arr[0].result()[0].data.evs)))
    for i, job in enumerate(jobs_arr):
        site_gauge_observable_matrix[i] = (1 - job.result()[0].data.evs)/2
    x_t_plot(site_gauge_observable_matrix)
