from qiskit.primitives.base.base_primitive_job import BasePrimitiveJob
from utils.plotting import convert_jobs_to_site_gauge_matrix
from qiskit_ibm_runtime.runtime_job import RuntimeJob
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

def x_t_plot(site_gauge_observable_matrix, cbar_range=None, nxticks=5, filepath=""):
    if isinstance(site_gauge_observable_matrix[0], BasePrimitiveJob) or isinstance(site_gauge_observable_matrix[0], RuntimeJob):
        site_gauge_observable_matrix = convert_jobs_to_site_gauge_matrix(site_gauge_observable_matrix)

    plt.rc("text", usetex=True)
    plt.rc("font", size=24, family="serif", weight="bold")

    fig, ax = plt.subplots(1, 1, figsize=[12, 8])

    aspect = site_gauge_observable_matrix.shape[0]/site_gauge_observable_matrix.shape[1]/15
    
    if cbar_range is None:
        plt.imshow(site_gauge_observable_matrix, cmap="inferno", aspect=aspect if aspect > 1/2 else 1/2)
    else:
        plt.imshow(site_gauge_observable_matrix, cmap="inferno", aspect=aspect if aspect > 1/2 else 1/2, vmax=cbar_range[1], vmin=cbar_range[0])
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
    plt.show()
    plt.rcdefaults()

def x_t_separated_plot(site_gauge_z_matrix, cbar_range_matter=None, cbar_range_gauge=None, nxticks=5, filepath=""):
    if isinstance(site_gauge_z_matrix[0], BasePrimitiveJob) or isinstance(site_gauge_z_matrix[0], RuntimeJob):
        site_gauge_z_matrix = convert_jobs_to_site_gauge_matrix(site_gauge_z_matrix)
    else:
        site_gauge_z_matrix = np.array(site_gauge_z_matrix)
    
    matter_occupation_matrix = (1 - site_gauge_z_matrix[:, 0::2])/2
    gauge_z_matrix = site_gauge_z_matrix[:, 1::2]

    plt.rc("text", usetex=True)
    plt.rc("font", size=24, family="serif", weight="bold")

    fig, ax = plt.subplots(1, 2, figsize=[12, 8], sharey=True, dpi=300)

    aspect = site_gauge_z_matrix.shape[0]/site_gauge_z_matrix.shape[1]/15

    if cbar_range_matter is None:
        imm = ax[0].imshow(matter_occupation_matrix, cmap="inferno", aspect=aspect if aspect > 1/2 else 1/2)
    else:
        imm = ax[0].imshow(matter_occupation_matrix, cmap="inferno", aspect=aspect if aspect > 1/2 else 1/2, vmax=cbar_range_matter[1], vmin=cbar_range_matter[0])
    
    if cbar_range_gauge is None:
        img = ax[1].imshow(gauge_z_matrix, cmap="bwr", aspect=aspect if aspect > 1/2 else 1/2)
    else:
        img = ax[1].imshow(gauge_z_matrix, cmap="bwr", aspect=aspect if aspect > 1/2 else 1/2, vmax=cbar_range_gauge[1], vmin=cbar_range_gauge[0])

    cbarm = plt.colorbar(imm, ax=ax[0])
    cbarm.ax.set_ylabel(r"$(1 - \langle Z \rangle)/2$", labelpad=10)

    cbarg = plt.colorbar(img, ax=ax[1])
    cbarg.ax.set_ylabel(r"$\langle Z \rangle$", labelpad=10)

    ax[0].set_xlabel(r"Matter sites")
    ax[1].set_xlabel(r"Gauge sites")
    ax[0].set_ylabel("step")

    ax[0].set_xticks(np.round(np.linspace(0, matter_occupation_matrix.shape[1]-1, nxticks)).astype(int), np.round(np.linspace(1, matter_occupation_matrix.shape[1], nxticks)).astype(int))
    gauge_ticks_location = np.round(np.linspace(0, gauge_z_matrix.shape[1]-1, nxticks)).astype(int)
    ax[1].set_xticks(gauge_ticks_location, [f"({g+1}, {g+2})" for g in gauge_ticks_location])

    # plt.suptitle(r"Particle \& Gauge occupation", horizontalalignment="center")

    # plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

def discrepancies_plot(exact_site_gauge_observable_matrix, approximated_site_gauge_observable_matrix, filepath=""):
    if isinstance(exact_site_gauge_observable_matrix[0], BasePrimitiveJob) or isinstance(exact_site_gauge_observable_matrix[0], RuntimeJob):
        exact_site_gauge_observable_matrix = convert_jobs_to_site_gauge_matrix(exact_site_gauge_observable_matrix)
    if isinstance(approximated_site_gauge_observable_matrix[0], BasePrimitiveJob) or isinstance(approximated_site_gauge_observable_matrix[0], RuntimeJob):
        approximated_site_gauge_observable_matrix = convert_jobs_to_site_gauge_matrix(approximated_site_gauge_observable_matrix)

    if exact_site_gauge_observable_matrix.shape != approximated_site_gauge_observable_matrix.shape:
        raise ValueError("exact_site_gauge_observable_matrix and approximated_site_gauge_observable matrix must be the same shape")

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
    plt.show()
    plt.rcdefaults()

def plot_single_qubit_discrepancies(exact_observable_matrix, approximated_observable_matrix, qubit_inds, x_plot=None, x_label=None, filepath=""):
    if isinstance(exact_observable_matrix[0], BasePrimitiveJob) or isinstance(exact_observable_matrix[0], RuntimeJob):
        exact_observable_matrix = convert_jobs_to_site_gauge_matrix(exact_observable_matrix)
    if isinstance(approximated_observable_matrix[0], BasePrimitiveJob) or isinstance(approximated_observable_matrix[0], RuntimeJob):
        approximated_observable_matrix = convert_jobs_to_site_gauge_matrix(approximated_observable_matrix)

    if exact_observable_matrix.shape != approximated_observable_matrix.shape:
        raise ValueError("exact_observable_matrix and approximated_observable matrix must be the same shape")

    plt.rc("text", usetex=True)
    plt.rc("font", size=24, family="serif", weight="bold")

    fig, ax = plt.subplots(1, 1, figsize=[9, 6])

    difference = np.abs(exact_observable_matrix - approximated_observable_matrix)

    cmap = plt.get_cmap("Set2")
    if x_plot is None:
        x_plot = np.arange(0, exact_observable_matrix.shape[0])
    for i, qbind in enumerate(qubit_inds):
        color = cmap((i % 8)/8 + 0.01)
        plt.plot(x_plot, difference[:, qbind], "o--", markersize=12, markeredgecolor="black", color=color, label=f"Qb: {qbind}")
    plt.xlabel("Step" if x_label is None else x_label, labelpad=10)
    plt.ylabel(r"$|\langle O \rangle_{\mathrm{Exact}} - \langle O \rangle_{\mathrm{Trotter}}|$", labelpad=10)
    plt.legend(prop={"size": 18})
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    plt.yscale("log")
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

def plot_average_discrepancy(exact_observable_matrix, approximated_observable_matrix, x_plot=None, x_label=None, filepath=""):
    if isinstance(exact_observable_matrix[0], BasePrimitiveJob) or isinstance(exact_observable_matrix[0], RuntimeJob):
        exact_observable_matrix = convert_jobs_to_site_gauge_matrix(exact_observable_matrix)
    if isinstance(approximated_observable_matrix[0], BasePrimitiveJob) or isinstance(approximated_observable_matrix[0], RuntimeJob):
        approximated_observable_matrix = convert_jobs_to_site_gauge_matrix(approximated_observable_matrix)

    if exact_observable_matrix.shape != approximated_observable_matrix.shape:
        raise ValueError("exact_observable_matrix and approximated_observable matrix must be the same shape")

    plt.rc("text", usetex=True)
    plt.rc("font", size=24, family="serif", weight="bold")

    fig, ax = plt.subplots(1, 1, figsize=[9, 6])

    difference = np.abs(exact_observable_matrix - approximated_observable_matrix)
    mean_difference = difference.mean(axis=1)
    
    cmap = plt.get_cmap("Set2")
    plt.plot(x_plot, mean_difference, "o--", markersize=12, markeredgecolor="black", color=cmap(0))
    plt.xlabel("Step" if x_label is None else x_label, labelpad=10)
    plt.ylabel(r"$|\langle O \rangle_{\mathrm{Exact}} - \langle O \rangle_{\mathrm{Trotter}}|$", labelpad=10)
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    plt.yscale("log")
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

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
            raise ValueError("x_arr must be the same length as samples_dicts")
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
    fig, ax = plt.subplots(figsize=[8, 5])
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
    plt.show()
    plt.rcdefaults()


def plot_average_separated(obs, ax, string):
    """
    plot average separated

    TODO description
    """
    times = range(np.asarray(obs).T.shape[0])
    matter_occupation_matrix = (1 - np.asarray(obs).T[:, 0::2])/2
    gauge_z_matrix = np.asarray(obs).T[:, 1::2]

    if string == "transverse":
        marker = "p"
    elif string == "longitudinal":
        marker = "^"
    av_occ = np.mean(matter_occupation_matrix, axis=1)
    av_gf = np.mean(gauge_z_matrix, axis=1)
    ax[0].plot(times, av_occ, marker, label=f"{string}")
    ax[0].set_ylabel(r"$(1 - \langle Z \rangle)/2$", labelpad=10)
    ax[0].set_xlabel(r"Matter sites")
    ax[1].plot(times, av_gf, marker, label=f"{string}")
    ax[1].set_ylabel(r"$\langle Z \rangle$", labelpad=10)
    ax[1].set_xlabel(r"Gauge sites")


def plot_string_observables( 
                            observable: np.ndarray, 
                            transverse: list=None, 
                            longitudinal: list=None,
                            cm: bool=False,
                            separated: bool=True,
                            sim_type: int=0,
                            obs_type: str="occ",
                            params: list=None,
                            std: np.ndarray=None

                            ):
    """
    plot string observables

    This function plots the values of the gauge fields <Z> or 
    the matter sites occupation (1 - <Z>)/2 for the qubits 
    longitudinal and/or transversal to the string.
    If both components are passed it plots the normalized averages of the
    observables in one plot. If cm (colormap) is true it returns two colormaps.
    These functionalities could be used to identify different physics modes 
    given by the time evolution.

    observable: np.ndarray - total array of observables
    transverse: list - list of qubit labels according to the layout chosen. By default None
    longitudinal: list - list of qubit labels according to the layout chosen. By default None
    cm: bool - colormap with every local observables. By default False, will plot averages
    sim_type: int - type of simulation chosen. Choose among: 0 - real, 1 - no odr, 
            2 - calibration circs odr, 3 - raw. By default 0
    obs_type: str - type of observable to plot. Choose among: "occ" occupation and "gf" gauge field. By default occupation
    params: list - parameters used during the time evolution. By default None
    std: np.ndarray - standard deviation of the observables. By default None

    """
    obs_t = None
    obs_l = None
    
    plt.rc("text", usetex=True)
    plt.rc("font", size=24, family="serif", weight="bold")
    plt.rc("figure", dpi=300)
    
    times = range(observable.shape[1])
    if separated:
        if transverse is not None:
            obs_t = [observable[sim_type,:,i] for i in transverse]
        if longitudinal is not None:
            obs_l = [observable[sim_type,:,i] for i in longitudinal]

        if cm:
            if obs_t is not None:
                print(f"\n -- tranverse:")
                x_t_separated_plot(np.asarray(obs_t).T)
            if obs_l is not None:
                print(f"\n -- longitudinal:")
                x_t_separated_plot(np.asarray(obs_l).T)
        
        else:
            fig, ax = plt.subplots(1, 2, figsize=[12, 8], sharey=True, dpi=300)

            if obs_t is not None:
                plot_average_separated(obs_t, ax, "transverse")
            if obs_l is not None:
                plot_average_separated(obs_l, ax, "longitudinal")

            
            ax[0].legend(fontsize=12)
            ax[1].legend(fontsize=12)
            ax[0].grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
            ax[1].grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
            plt.suptitle("Matter $\&$ Gauge sites of the transverse and longitudinal strings")
            plt.show()
            
    else:
        if obs_type == "occ":
            observable = (1 - observable[:,:]) / 2

        if transverse is not None:
            obs_t = [observable[sim_type,:,i] for i in transverse]
        if longitudinal is not None:
            obs_l = [observable[sim_type,:,i] for i in longitudinal]
            
        if cm:
            if obs_t is not None:
                print(f"\n -- tranverse:")
                x_t_plot(np.asarray(obs_t).T)
            if obs_l is not None:
                print(f"\n -- longitudinal:")
                x_t_plot(np.asarray(obs_l).T)

        else:
            if obs_t is not None:
                av_obs_t = np.mean(obs_t, axis=0)
                plt.plot(times, av_obs_t, 'p', label="Transverse")
            if obs_l is not None:
                av_obs_l = np.mean(obs_l, axis=0)
                plt.plot(times, av_obs_l, '^', label="Longitudinal")
            
            plt.title("Gauge sites of the transverse and longitudinal strings")
            plt.xlabel("Circuit index")
            plt.ylabel("$\\frac{1}{N}\\sum_{x} \\langle Z(x) \\rangle$")
            plt.legend(fontsize=12)
            plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
            plt.show()