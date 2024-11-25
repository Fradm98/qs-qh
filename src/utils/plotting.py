from qiskit.primitives.base.base_primitive_job import BasePrimitiveJob
from utils.postselection import measure_diagonal_observables
from qiskit.primitives.primitive_job import PrimitiveJob
from utils.circs import depth2qb
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import sys

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

def observable_depth_mitigation_comparison_plot(plot_dict, x_plot, y_lim=[], x_label="", y_label="", colors=None, markers=["o", "^", "s", "X", "v"], regression=False, resolution=100, filepath=""):
    # plot_dict:
    #    {
    #        "<observable_label>": {
    #            "<subindex (Mitigation label)>": [<observable_list>]
    #        }
    #    }

    plt.rc("text", usetex=True)
    plt.rc("font", size=24, family="serif", weight="bold")

    fig, ax = plt.subplots(figsize=[9, 6])
    reg_func = lambda x, a: np.exp(-a*x)
    x_regression = np.linspace(x_plot[0], x_plot[-1], resolution)
    cmap = plt.get_cmap("Set2")

    min_observable = sys.float_info.max
    max_observable = sys.float_info.min

    for i, observable_label in enumerate(plot_dict.keys()):
        subindices = list(plot_dict[observable_label].keys())
        observable_arrays = list(plot_dict[observable_label].values())
        
        if len(obs_len := np.unique([len(oarr) for oarr in observable_arrays])) > 1:
            raise ValueError("Not all observable lists have the same length")
        if len(x_plot) != obs_len:
            raise ValueError("x_plot must have the same length as all observable arrays")
        
        if colors is None:
            color = cmap((i % 8)/8 + 0.01)
        else:
            color = colors[i]

        for j, (subindex, observable) in enumerate(zip(subindices[::-1], observable_arrays[::-1])):
            if np.max(observable) > max_observable:
                max_observable = np.max(observable)
            if np.min(observable) < min_observable:
                min_observable = np.min(observable)
            this_color = color if j == len(observable_arrays)-1 else tuple(0.6*c if i < 3 else c for i, c in enumerate(color))
            this_markersize = np.arange(11-len(observable_arrays), 11)[j]
            if regression:
                popt, _ = sp.optimize.curve_fit(reg_func, x_regression, observable, p0=[1])
                plt.plot(x_regression, reg_func(x_regression, *popt), "--", linestyle="dashed", color=this_color, label=f"$a={popt[0]:.03f}$\n")
            plt.plot(x_plot, observable, markers[len(observable_arrays)-1-j], markersize=this_markersize, markeredgecolor="black", color=this_color, label=r"$\langle %s \rangle_{\mathrm{%s}}$" % (observable_label, subindex), zorder=5)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if y_lim:
        plt.ylim(y_lim)
    else:
        plt.ylim([min_observable - 0.05*(max_observable - min_observable), max_observable + 0.05*(max_observable - min_observable)])
    plt.legend(prop={"size": 18})
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()