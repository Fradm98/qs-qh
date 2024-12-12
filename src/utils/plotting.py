from utils.postselection import diagonal_operators_check, get_layout_state
from qiskit.primitives.base.base_primitive_job import BasePrimitiveJob
from utils.postselection import measure_diagonal_observables
from qiskit.primitives.primitive_job import PrimitiveJob
from utils.circs import depth2qb
import matplotlib.pyplot as plt
from pymatching import Matching
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
    return site_gauge_observable_matrix

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
    reg_func = lambda x, a: a**x
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
                unique_x = np.unique(x_plot)
                unique_y = np.zeros_like(unique_x, dtype=float)
                errors = np.zeros_like(unique_x, dtype=float)
                for i, x in enumerate(unique_x):
                    this_x_mask = np.equal(x, x_plot)
                    this_ys = observable[this_x_mask]
                    unique_y[i] = np.mean(this_ys)
                    errors[i] = np.std(this_ys)
                popt, _ = sp.optimize.curve_fit(reg_func, unique_x, unique_y, p0=[0.5])
                plt.plot(x_regression, reg_func(x_regression, *popt), "--", linestyle="dashed", color=this_color, label=f"$a={popt[0]:.03f}$\n")
            plt.plot(x_plot, observable, markers[(len(observable_arrays)-1-j) % 5], markersize=this_markersize, markeredgecolor="black", color=this_color, label=r"$\langle %s \rangle_{\mathrm{%s}}$" % (observable_label, subindex), zorder=5)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if y_lim:
        plt.ylim(y_lim)
    else:
        plt.ylim([min_observable - 0.05*(max_observable - min_observable), max_observable + 0.05*(max_observable - min_observable)])
    plt.legend(prop={"size": 18})
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    if regression: plt.title("Regression: $a^x$")
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

def plot_nflips_comparison(samples_dicts, postselection_operators, g_arr, circs_layout=None, filepath=""):
    plt.rc("text", usetex=True)
    plt.rc("font", size=20, family="serif", weight="bold")

    if len(samples_dicts) != 2:
        raise ValueError("Only two sample dicts can be compared")

    all_diagonal, basis = diagonal_operators_check(postselection_operators)
    if not all_diagonal:
        raise ValueError("Only supports diagonal postselection operators in some basis")

    postselection_mask = np.array([[int(str(opel) == basis) for opel in operator] for operator in postselection_operators])
    decoder = Matching(postselection_mask)
    flip_counts = []
    total_nflips = np.zeros(len(samples_dicts), dtype=int)
    max_nflips = 0
    for i, samples_dict in enumerate(samples_dicts):
        states = np.array([[int(c) for c in string] for string in samples_dict.keys()])
        state_counts = np.array(list(samples_dict.values()))
        if circs_layout is not None:
            states = get_layout_state(states, circs_layout[i])
        states_syndromes = (states @ postselection_mask.T) % 2
        predicted_flips = decoder.decode_batch(states_syndromes)
        nflips = predicted_flips.sum(axis=1)
        unique_nflips = np.unique(nflips)
        this_flip_counts = {}
        this_total_nflips = 0
        for nf in unique_nflips:
            this_counts = np.sum(state_counts[np.equal(nflips, nf)])
            this_flip_counts[nf] = this_counts
            this_total_nflips += this_counts
        total_nflips[i] = this_total_nflips
        this_max_nflips = max(this_flip_counts.keys())
        if this_max_nflips > max_nflips: max_nflips = this_max_nflips
        flip_counts.append(this_flip_counts)

    if len(np.unique(total_nflips)) > 1:
        print("WARNING: Unfair comparison, different number of total samples")
        
    cmap = plt.get_cmap("Set2")
    x_plot = np.arange(max_nflips+1)
    fig, ax = plt.subplots(figsize=[9, 6])
    bars = []
    ys = []
    for i, fc in enumerate(flip_counts):
        color = cmap((i % 8)/8 + 0.01)
        y_plot = np.array([fc.get(nf, np.nan) for nf in x_plot])
        try:
            g_str = f"$g = {float(g_arr[i]):.02f}$"
        except TypeError:
            g_str = f"$g$ = {str(g_arr[i])}"
        width = 0.4 if i == 0 else -0.4
        this_bars = plt.bar(x_plot, y_plot, color=color, zorder=3, label=g_str, align="edge", width=width, edgecolor="black", linewidth=0.7)
        ys.append(y_plot)
        bars.append(this_bars)

    for i, (b0, b1) in enumerate(zip(bars[0], bars[1])):
        if ys[0][i] > ys[1][i]:
            b0.zorder = 3
            b1.zorder = 5
        else:
            b0.zorder = 5
            b1.zorder = 3

    plt.xticks(x_plot)
    plt.xlabel(r"\# detected flips")
    plt.ylabel("Counts")
    plt.title(f"Total samples: {np.max(total_nflips)}", pad=8)
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    plt.legend()
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()