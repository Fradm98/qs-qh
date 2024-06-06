import matplotlib.pyplot as plt
import numpy as np

def x_t_plot(site_guge_observable_matrix):
    plt.rc("text", usetex=True)
    plt.rc("font", size=24, family="serif", weight="bold")

    fig, ax = plt.subplots(1, 1, figsize=[8, 8*site_guge_observable_matrix.shape[0]/site_guge_observable_matrix.shape[1]])

    plt.imshow(site_guge_observable_matrix, cmap="inferno")
    plt.xlabel(r"Matter (Even) \& Gauge (Odd) sites")
    plt.ylabel("t")