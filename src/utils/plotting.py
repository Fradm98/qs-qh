from utils.postselection import get_recovered_postselected_samples_dict, combine_samples_dicts, get_logical_samples_dict
from utils.postselection import diagonal_operators_check, get_layout_state
from qiskit.primitives.base.base_primitive_job import BasePrimitiveJob
from qiskit.primitives.primitive_job import PrimitiveJob
from utils.circs import depth2qb
import matplotlib.pyplot as plt
from pymatching import Matching
import scipy as sp
import numpy as np
import sys
import os

def luminance(*rgb):
    """Calcula la luminancia relativa de un color."""
    rgb = list(rgb)
    for i in range(3):
        if rgb[i] <= 0.03928:
            rgb[i] = rgb[i] / 12.92
        else:
            rgb[i] = ((rgb[i] + 0.055) / 1.055) ** 2.4
    return rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722

def contrast_ratio(l1, l2):
    """Calcula la relación de contraste entre dos luminancias."""
    if l1 > l2:
        return (l1 + 0.05) / (l2 + 0.05)
    else:
        return (l2 + 0.05) / (l1 + 0.05)

def best_text_color(*rgb):
    """Determina si el texto blanco o negro se verá mejor sobre un color dado."""
    lum = luminance(*rgb)
    contrast_with_white = contrast_ratio(lum, luminance(1, 1, 1))
    contrast_with_black = contrast_ratio(lum, luminance(0, 0, 0))
    
    if contrast_with_white > contrast_with_black:
        return "white"
    else:
        return "black"

def convert_jobs_to_site_gauge_matrix(jobs_arr):
    printed_warning_header = False
    first_success_index = 0
    while first_success_index < len(jobs_arr):
        try:
            first_success_job = jobs_arr[first_success_index]
            if not first_success_job.in_final_state() and type(first_success_job) is not PrimitiveJob:
                raise TimeoutError("Some jobs are still to be finished")
            else:
                first_success_result = first_success_job.result()[0].data.evs
            site_gauge_observable_matrix = np.full((len(jobs_arr), len(first_success_result)), np.nan)
            break
        except:
            if not printed_warning_header: 
                print("WARNING: Some jobs failed, the plot will be incomplete\nFAILED JOBS\nIndex | ID")
                printed_warning_header = True
            print(f"{str(first_success_index).ljust(6)}| {jobs_arr[first_success_index].job_id()}")
            first_success_index += 1
            continue
    # site_gauge_observable_matrix[first_success_index] = (1 - first_success_result[::-1])/2
    site_gauge_observable_matrix[first_success_index] = first_success_result
    for i, job in enumerate(jobs_arr[first_success_index::], start=first_success_index):
        try:
            if not job.in_final_state() and type(first_success_job) is not PrimitiveJob:
                raise TimeoutError("Some jobs are still to be finished")
            else:
                this_job_result = job.result()[0].data.evs
            site_gauge_observable_matrix[i] = this_job_result
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

    if os.path.splitext(filepath)[1] == ".npy":
        np.save(filepath, site_gauge_observable_matrix)
    else:
        np.savetxt(filepath, site_gauge_observable_matrix, header=header)

def load_site_gauge_observable_matrix(filepath):
    if os.path.splitext(filepath)[1] == ".npy":
        return np.load(filepath)
    else:
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

def plot_nflips_comparison(samples_dicts, postselection_operators, g_arr, circs_layout=None, legend=True, filepath=""):
    plt.rc("text", usetex=True)
    plt.rc("font", size=20, family="serif", weight="bold")

    if len(samples_dicts) > 2:
        raise ValueError("At most two sample dicts can be compared")

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

    # for i, (b0, b1) in enumerate(zip(bars[0], bars[1])):
    #     if ys[0][i] > ys[1][i]:
    #         b0.zorder = 3
    #         b1.zorder = 5
    #     else:
    #         b0.zorder = 5
    #         b1.zorder = 3

    plt.xticks(x_plot)
    plt.xlabel(r"\# detected flips")
    plt.ylabel("Counts")
    plt.title(f"Total samples: {np.max(total_nflips)}", pad=8)
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    if legend: plt.legend()
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

def plot_circuit_arr_depth(circuit_arr, x_ticks=None, x_label=None, filepath=""):
    plt.rc("text", usetex=True)
    plt.rc("font", size=20, family="serif", weight="bold")

    fig, ax = plt.subplots(figsize=[9, 6])
    
    if x_ticks is None:
        x_ticks = np.arange(1, len(circuit_arr)+1)
    y = [depth2qb(circ) for circ in circuit_arr]
    color = plt.get_cmap("Set2")((1 % 8)/8 + 0.01)
    plt.plot(x_ticks, y, "o--", markersize=12, markeredgecolor="black", color=color)
    if x_label is None:
        plt.xlabel("Circuit index", labelpad=10)
    else:
        plt.xlabel(x_label, labelpad=10)
    plt.ylabel("Depth")
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

def plot_individual_qubits_observable_evolution(observable_matrix, observable_confidence_matrix=None, function_over=None, ylabel=None, cmap=None, x_ticks=None, x_label=None, y_lim=None, fmt="", funclabels=None, title="", legend=True, legend_fontsize=None, data_linewidth=1.5, filepath=None):
    observable_matrix = np.array(observable_matrix)
    if observable_confidence_matrix is not None:
        observable_confidence_matrix = np.array(observable_confidence_matrix)
        if observable_matrix.shape != observable_confidence_matrix.shape[:-1]:
            raise ValueError("Observable matrix and standard deviation matrix must have the same shape")
    if funclabels is not None:
        try:
            if len(funclabels) != (observable_matrix.shape[1] + (1 if function_over is not None else 0)):
                raise ValueError("funclabels must have the same length as the number of qubits")
        except TypeError:
            raise ValueError("funclabels must have the same length as the number of qubits")
    
    if cmap is None:
        cmap = plt.get_cmap("Set2")
    plt.rc("text", usetex=True)
    plt.rc("font", size=20, family="serif", weight="bold")
    fig, ax = plt.subplots(figsize=[9, 6])
    if x_ticks is not None:
        if len(x_ticks.shape) == 1:
            plt.xticks(x_ticks)
        else:
            plt.xticks(*x_ticks)
        x = np.linspace(x_ticks[0], x_ticks[-1], observable_matrix.shape[0])
    else:
        x = np.arange(observable_matrix.shape[0])
    for i in range(observable_matrix.shape[1]):
        color = cmap((i % 8))
        y = observable_matrix[:, i]
        if observable_confidence_matrix is not None:
            upper_error_bound_mask = np.less(y.reshape((len(y), 1)), observable_confidence_matrix[:, i, :])
            lower_error_bound_mask = np.greater(y.reshape((len(y), 1)), observable_confidence_matrix[:, i, :])
            consistency_mask = ~np.logical_xor(np.sum(upper_error_bound_mask, axis=1) == 0, np.sum(lower_error_bound_mask, axis=1) == 0)
            upper_error = observable_confidence_matrix[:, i, 1][consistency_mask] - y[consistency_mask]
            lower_error = y[consistency_mask] - observable_confidence_matrix[:, i, 0][consistency_mask]
            yerr = np.zeros_like(upper_error_bound_mask, dtype=float)
            yerr[consistency_mask, 0] = lower_error
            yerr[consistency_mask, 1] = upper_error
            # if i == 4:
                # plt.errorbar(x, y, yerr=yerr.T, fmt="o--" if fmt is None else fmt, markersize=8, linewidth=data_linewidth, elinewidth=2, markeredgecolor="black", color=color, label=funclabels[i] if funclabels is not None else f"Qubit {i}", capsize=5, capthick=2)
            # else:
                # plt.errorbar(x[::2], y[::2], yerr=yerr[::2, :].T, fmt="o--" if fmt is None else fmt, markersize=8, linewidth=data_linewidth, elinewidth=1.5, markeredgecolor="black", color=color, label=funclabels[i] if funclabels is not None else f"Qubit {i}", capsize=5, capthick=2)
                # plt.errorbar(x, y, yerr=yerr.T, fmt="o--" if fmt is None else fmt, markersize=8, linewidth=data_linewidth, elinewidth=1.5, markeredgecolor="black", color=color, label=funclabels[i] if funclabels is not None else f"Qubit {i}", capsize=5, capthick=2)
            plt.fill_between(x, observable_confidence_matrix[:, i, 0], observable_confidence_matrix[:, i, 1], color=cmap(i), alpha=0.3)
            plt.plot(x, observable_matrix[:, i], "o", markersize=8, markeredgecolor="black", color=color)
        else:
            plt.plot(x, y, "o--" if fmt is None else fmt, markersize=8, linewidth=data_linewidth, markeredgecolor="black", color=color, label=funclabels[i] if funclabels is not None else f"Qubit {i}")
    if function_over is not None:
        if callable(function_over):
            plt.plot(x, function_over(x), "-", color=cmap(((i+1) % 8)), label=funclabels[i+1] if funclabels is not None else f"Function", linewidth=2.5)
        else:
            x_func = np.linspace(x[0], x[-1], len(function_over))
            plt.plot(x_func, function_over, "-", color=cmap(((i+1) % 8)), label=funclabels[i+1] if funclabels is not None else f"Function", linewidth=2.5)
    plt.xlabel(x_label)
    plt.ylabel(ylabel, labelpad=10)
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    if y_lim == "data":
        min_data = np.nanmin(observable_matrix)
        max_data = np.nanmax(observable_matrix)
        plt.ylim([min_data - 0.05*(max_data - min_data), max_data + 0.05*(max_data - min_data)])
    elif y_lim is not None:
        plt.ylim(y_lim)
    if title:
        plt.title(title, pad=10)
    plt.tight_layout()
    if legend: plt.legend(loc="best", fontsize=16 if legend_fontsize is None else legend_fontsize)
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

def plot_average_qubits_observable_evolution(observable_matrix, average_qubits, observable_confidence_matrix=None, cmap=None, x_ticks=None, x_label=None, y_lim=None, y_ticks=None, ylabel=None, markers=None, funclabels=None, title="", legend=True, filepath=None):
    observable_matrix = np.array(observable_matrix)
    if observable_confidence_matrix is not None:
        observable_confidence_matrix = np.array(observable_confidence_matrix)
        if observable_matrix.shape != observable_confidence_matrix.shape[:-1]:
            raise ValueError("Observable matrix and standard deviation matrix must have the same shape")
    
    average_observable_matrix = np.zeros((observable_matrix.shape[0], len(average_qubits)))
    if observable_confidence_matrix is not None:
        average_observable_confidence_matrix = np.zeros((observable_matrix.shape[0], len(average_qubits), 2))
    for i, average_qubits_arr in enumerate(average_qubits):
        this_reduced_observable_matrix = observable_matrix[:, average_qubits_arr]
        average_observable_matrix[:, i] = np.nanmean(this_reduced_observable_matrix, axis=1)
        if observable_confidence_matrix is not None:
            this_reduced_observable_std_matrix = observable_confidence_matrix[:, average_qubits_arr, :]
            average_observable_confidence_matrix[:, i, :] = np.nanmean(this_reduced_observable_std_matrix, axis=1)
    
    if cmap is None:
        cmap = plt.get_cmap("Set2")
    plt.rc("text", usetex=True)
    plt.rc("font", size=20, family="serif", weight="bold")
    fig, ax = plt.subplots(figsize=[9, 6])
    if x_ticks is not None:
        if len(x_ticks.shape) == 1:
            plt.xticks(x_ticks)
        else:
            plt.xticks(*x_ticks)
        x = np.linspace(x_ticks[0], x_ticks[-1], observable_matrix.shape[0])
    else:
        x = np.arange(observable_matrix.shape[0])
    if funclabels is None:
        funclabels = [f"Line {i}" for i in range(len(average_qubits))]
    for i in range(len(average_qubits)):
        color = cmap((i % 8))
        y = average_observable_matrix[:, i]
        if observable_confidence_matrix is not None:
            minus_error = y - average_observable_confidence_matrix[:, i, 1]
            plus_error = average_observable_confidence_matrix[:, i, 0] - y
            yerr = np.array([minus_error, plus_error])
            plt.errorbar(x, y, yerr=yerr, fmt="--", marker=markers[i] if markers is not None else "o", markersize=10, elinewidth=1.5, markeredgecolor="black", color=color, label=funclabels[i])
        else:
            plt.plot(x, y, "--", marker=markers[i] if markers is not None else "o", markersize=10, markeredgecolor="black", color=color, label=funclabels[i])
    plt.xlabel(x_label)
    plt.ylabel(ylabel, labelpad=10)
    if y_ticks is not None:
        plt.yticks(y_ticks)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    if legend: plt.legend(loc="best", fontsize=16)
    if title:
        plt.title(title, pad=10)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

def plot_discarded_samples(samples_dicts, postselection_operators, flip_thresholds, circs_layouts=None, cmap=None, x_ticks=None, x_ticks_upper_func=None, x_label=None, x_label_upper=None, y_ticks=None, markers=None, regression=False, title="", legend=False, filepath=""):
    try:
        odr_samples_dicts = samples_dicts[1]
        real_start_index = odr_samples_dicts.shape[1]
    except IndexError:
        real_start_index = 0
    real_samples_dicts = samples_dicts[0]
    if circs_layouts is not None and real_samples_dicts.shape != (circs_layouts.shape[0], circs_layouts.shape[1]-real_start_index):
        raise ValueError("samples_dicts and circs_layouts do not have appropiate shapes")
    accepted_samples_ratio = np.zeros((len(flip_thresholds), real_samples_dicts.shape[1]))
    for i, flip_threshold in enumerate(flip_thresholds):
        for time_ind in range(real_samples_dicts.shape[1]):
            all_layouts_samples = real_samples_dicts[:, time_ind]
            if circs_layouts is not None:
                to_combine_samples = [get_logical_samples_dict(all_layouts_samples[j], circs_layouts[j, time_ind+real_start_index]) for j in range(len(all_layouts_samples))]
            else:
                to_combine_samples = all_layouts_samples
            combined_samples = combine_samples_dicts(*to_combine_samples)
            postselected_combined_samples = get_recovered_postselected_samples_dict(combined_samples, postselection_operators, flip_threshold=flip_threshold)
            accepted_samples_ratio[i, time_ind] = sum(postselected_combined_samples.values())/sum(combined_samples.values())
    
    if cmap is None:
        cmap = plt.get_cmap("Set2")
    plt.rc("text", usetex=True)
    plt.rc("font", size=20, family="serif", weight="bold")
    fig, ax = plt.subplots(figsize=[9, 6])
    if x_ticks is not None:
        x = np.linspace(x_ticks[0], x_ticks[-1], real_samples_dicts.shape[1])
    else:
        x = np.arange(samples_dicts.shape[1])
    for i, flip_threshold in enumerate(flip_thresholds):
        color = cmap((i % 8))
        plt.plot(x, accepted_samples_ratio[i], marker=markers[i] if markers is not None else "o", markersize=10, markeredgecolor="black", color=color, label=f"Max. {flip_threshold} flips")

    if regression:
        def reg_func(x, a, b):
            return b*np.exp(-a*x)
        x_regression = np.linspace(x[0], x[-1], 100)
        for i, flip_threshold in enumerate(flip_thresholds):
            popt, _ = sp.optimize.curve_fit(reg_func, x, accepted_samples_ratio[i], p0=[1, 1])
            plt.plot(x_regression, reg_func(x_regression, *popt), "--", color=cmap((i % 8)), label=rf"$a = {popt[0]}, b = {popt[1]}$")
    
    if x_label is not None:
        plt.xlabel(x_label)
    elif x_label is None and x_ticks is None:
        plt.xlabel("Job index", labelpad=10)
    if x_ticks_upper_func is not None:
        upper_xticks = ax.twiny()
        upper_xticks.set_xticks(x_ticks)
        upper_xticks.set_xticklabels(x_ticks_upper_func(x_ticks))
        if x_label_upper is not None:
            upper_xticks.set_xlabel(x_label_upper, labelpad=10)
    if y_ticks is not None:
        plt.yticks(y_ticks)
    plt.ylabel(r"$N_{\mathrm{post}}/N_{\mathrm{total}}$", labelpad=10)
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    if legend: plt.legend(loc="best", fontsize=16)
    if title:
        plt.title(title, pad=10)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

def plot_mean_deviations(observable_matrices_1, observable_matrices_2, x_ticks=None, x_label=None, y_lim=None, labels=None, cmap=None, title="", filepath=None):
    observable_matrices_1 = np.array(observable_matrices_1)
    observable_matrices_2 = np.array(observable_matrices_2)

    if observable_matrices_1.shape != observable_matrices_2.shape:
        raise ValueError("Observable matrices must have the same shape")
    
    if len(observable_matrices_1.shape) == 2:
        observable_matrices_1 = observable_matrices_1[None]
        observable_matrices_2 = observable_matrices_2[None]

    if labels is not None:
        if len(labels) != observable_matrices_1.shape[0]:
            raise ValueError("labels must have the same length as number of observable")
    else:
        labels = [f"Params ${i}$" for i in range(observable_matrices_1.shape[0])]
    
    plt.rc("text", usetex=True)
    plt.rc("font", size=20, family="serif", weight="bold")
    if cmap is None:
        cmap = plt.get_cmap("Set2")
    
    mean_deviations = np.nanmean(np.abs(observable_matrices_1 - observable_matrices_2), axis=-1)
    x = np.arange(mean_deviations.shape[1])

    fig, ax = plt.subplots(figsize=[9, 6])
    for i, mean_deviation in enumerate(mean_deviations):
        color = cmap((i % 8)/8 + 0.01)
        plt.plot(x, mean_deviation, "o--", markersize=12, markeredgecolor="black", color=color, label=labels[i])
    if x_ticks is not None:
        plt.xticks(*x_ticks)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.ylabel(r"$| \langle O \rangle_1 - \langle O \rangle_2 |$", labelpad=10)
    if title:
        plt.title(title)
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    plt.tight_layout()
    plt.legend(loc="best", fontsize=12)
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

def plot_max_deviation(observable_matrices_1, observable_matrices_2, x_ticks=None, x_label=None, y_lim=None, labels=None, cmap=None, title="", filepath=None):
    observable_matrices_1 = np.array(observable_matrices_1)
    observable_matrices_2 = np.array(observable_matrices_2)

    if observable_matrices_1.shape != observable_matrices_2.shape:
        raise ValueError("Observable matrices must have the same shape")
    
    if len(observable_matrices_1.shape) == 2:
        observable_matrices_1 = observable_matrices_1[None]
        observable_matrices_2 = observable_matrices_2[None]

    if labels is not None:
        if len(labels) != observable_matrices_1.shape[0]:
            raise ValueError("labels must have the same length as number of observable")
    else:
        labels = [f"Params ${i}$" for i in range(observable_matrices_1.shape[0])]
    
    plt.rc("text", usetex=True)
    plt.rc("font", size=20, family="serif", weight="bold")
    if cmap is None:
        cmap = plt.get_cmap("Set2")
    
    max_deviations = np.nanmax((np.abs(observable_matrices_1 - observable_matrices_2)), axis=1)
    x = np.arange(len(max_deviations))

    fig, ax = plt.subplots(figsize=[9, 6])
    for i, max_deviation in enumerate(max_deviations):
        color = cmap((i % 8)/8 + 0.01)
        plt.plot(x, max_deviations, "o--", markersize=12, markeredgecolor="black", color=color, label=labels[i])
    
    if x_ticks is not None:
        plt.xticks(*x_ticks)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.ylabel(r"$\max_k | \langle O_k \rangle_1 - \langle O_k \rangle_2 |$", labelpad=10)
    if title:
        plt.title(title)
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    plt.tight_layout()
    plt.legend(loc="best", fontsize=12)
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

def plot_individual_qubits_htn_comparison(observable_matrix_hardware, observable_matrix_tn, qubit_indices, observable_confidence_hardware=None, x_ticks=None, x_label=None, y_label=None, y_lim=None, labels=None, cmap=None, title="", legend=None, filepath=None):
    observable_matrix_hardware = np.array(observable_matrix_hardware)[:, qubit_indices]
    observable_matrix_tn = np.array(observable_matrix_tn)[:, qubit_indices]
    if observable_confidence_hardware is not None:
        observable_confidence_hardware = np.array(observable_confidence_hardware)[:, qubit_indices, :]
        if observable_matrix_hardware.shape != observable_confidence_hardware.shape[:-1]:
            raise ValueError("Observable matrix and standard deviation matrix must have the same shape")
    
    if labels is not None:
        if len(labels) != observable_matrix_hardware.shape[1]:
            raise ValueError("labels must have the same length as number of observable")
    
    if cmap is None:
        cmap = plt.get_cmap("Set2")
    plt.rc("text", usetex=True)
    plt.rc("font", size=22, family="serif", weight="bold")
    fig, ax = plt.subplots(figsize=[9, 6])

    if x_ticks is not None:
        if len(x_ticks.shape) == 1:
            plt.xticks(x_ticks)
        else:
            plt.xticks(*x_ticks)
        x_hardware = np.linspace(x_ticks[0], x_ticks[-1], observable_matrix_hardware.shape[0])
        x_tn = np.linspace(x_ticks[0], x_ticks[-1], observable_matrix_tn.shape[0])
    else:
        x_hardware = np.arange(observable_matrix_hardware.shape[0])
        x_tn = np.arange(observable_matrix_tn.shape[1])

    if labels is None:
        labels = [f"$O_{i}$" for i in range(observable_matrix_hardware.shape[1])]

    hlines = []
    for i in range(observable_matrix_hardware.shape[1]):
        color = cmap((i % 8))
        plt.plot(x_tn, observable_matrix_tn[:, i], "-", markersize=10, markeredgecolor="black", color=color, label=labels[i], linewidth=2.5)
        yh = observable_matrix_hardware[:, i]
        if observable_confidence_hardware is not None:
            # upper_error_bound_mask = np.less(observable_matrix_hardware[:, i].reshape((-1, 1)), observable_confidence_hardware[:, i, :])
            # lower_error_bound_mask = np.greater(observable_matrix_hardware[:, i].reshape((-1, 1)), observable_confidence_hardware[:, i, :])
            # upper_error = observable_confidence_hardware[:, i, :][upper_error_bound_mask] - observable_matrix_hardware[:, i]
            # lower_error = observable_matrix_hardware[:, i] - observable_confidence_hardware[:, i, :][lower_error_bound_mask]
            # yerr = np.array([lower_error, upper_error])
            upper_error_bound_mask = np.less(yh.reshape((len(yh), 1)), observable_confidence_hardware[:, i, :])
            lower_error_bound_mask = np.greater(yh.reshape((len(yh), 1)), observable_confidence_hardware[:, i, :])
            consistency_mask = ~np.logical_xor(np.sum(upper_error_bound_mask, axis=1) == 0, np.sum(lower_error_bound_mask, axis=1) == 0)
            upper_error = observable_confidence_hardware[:, i, 1][consistency_mask] - yh[consistency_mask]
            lower_error = yh[consistency_mask] - observable_confidence_hardware[:, i, 0][consistency_mask]
            yerr = np.zeros_like(upper_error_bound_mask, dtype=float)
            yerr[consistency_mask, 0] = lower_error
            yerr[consistency_mask, 1] = upper_error
            plt.fill_between(x_hardware, observable_confidence_hardware[:, i, 0], observable_confidence_hardware[:, i, 1], color=color, alpha=0.3)
            hlines.append(plt.plot(x_hardware, observable_matrix_hardware[:, i], "o", markersize=8, markeredgecolor="black", color=color, label=labels[i]))
            # hlines.append(plt.errorbar(x_hardware, observable_matrix_hardware[:, i], yerr=yerr.T, fmt="o", markersize=10, elinewidth=2, markeredgecolor="black", color=color, label=labels[i], capsize=5, capthick=2))
        else:
            hlines.append(plt.plot(x_hardware, observable_matrix_hardware[:, i], "o", markersize=10,  markeredgecolor="black", color=color, label=labels[i]))
    
    plt.xlabel(x_label)
    plt.ylabel(y_label, labelpad=10)
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    if y_lim == "data":
        min_data = np.nanmin(observable_matrix_hardware)
        max_data = np.nanmax(observable_matrix_hardware)
        plt.ylim([min_data - 0.05*(max_data - min_data), max_data + 0.05*(max_data - min_data)])
    elif y_lim is not None:
        plt.ylim(y_lim)
    if title:
        plt.title(title, pad=10)
    plt.tight_layout()
    if legend:
        plt.legend(handles=[plt.plot([], "o-", color=cmap(i % 8))[0] for i in range(observable_confidence_hardware.shape[1])], labels=[line.get_label() for line in hlines], loc="upper right", fontsize=16)
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

def plot_discarded_samples(postselected_samples_dicts, samples_dicts, x_ticks=None, x_label=None, labels=None, cmap=None, title="", filepath=None):
    postselected_samples_dicts = np.array(postselected_samples_dicts, dtype=object)
    samples_dicts = np.array(samples_dicts, dtype=object)

    if postselected_samples_dicts.shape != samples_dicts.shape:
        raise ValueError("postselected_samples_dicts and samples_dicts must have the same shape")
    
    if len(postselected_samples_dicts.shape) == 1:
        postselected_samples_dicts = postselected_samples_dicts[None]
        samples_dicts = samples_dicts[None]
    elif len(postselected_samples_dicts.shape) > 2:
        raise ValueError("postselected_samples_dicts and samples_dicts must be 1D or 2D arrays")
    
    if labels is not None:
        if len(labels) != postselected_samples_dicts.shape[0]:
            raise ValueError("labels must have the same length as number of observable")
    
    if cmap is None:
        cmap = plt.get_cmap("Set2")
    plt.rc("text", usetex=True)
    plt.rc("font", size=20, family="serif", weight="bold")
    fig, ax = plt.subplots(figsize=[9, 6])

    if x_ticks is not None:
        if len(np.array(x_ticks).shape) > 1:
            raise ValueError("x_ticks must be 1D array")
        x = np.linspace(x_ticks[0], x_ticks[-1], postselected_samples_dicts.shape[1])
    else:
        x = np.arange(postselected_samples_dicts.shape[1])

    for i in range(postselected_samples_dicts.shape[0]):
        color = cmap((i % 8)/8)
        nsamples_og = np.array([sum(sd.values()) for sd in samples_dicts[i]])
        nsamples_ps = np.array([sum(sd.values()) for sd in postselected_samples_dicts[i]])
        ratio = nsamples_ps/nsamples_og
        plt.plot(x, ratio, "o--", markersize=12, markeredgecolor="black", color=color, label=labels[i] if labels is not None else None)
    
    plt.xlabel(x_label)
    plt.ylabel(r"$n_{\mathrm{post}}/n_{\mathrm{raw}}$", labelpad=10)
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    if title:
        plt.title(title, pad=10)
    plt.tight_layout()
    if labels is not None:
        plt.legend(loc="best", fontsize=12)
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

def plot_string_configuration_evolution(samples_dicts, strings_qubits, x_ticks=None, x_label=None, cmap=None, labels=None, title="", filepath=None):
    samples_dicts = np.array(samples_dicts, dtype=object)

    if len(samples_dicts.shape) > 1:
        raise ValueError("samples_dicts must be 1D array")
    if not isinstance(strings_qubits[0], (list, tuple)):
        strings_qubits = [strings_qubits]
    
    if cmap is None:
        cmap = plt.get_cmap("Set2")

    if x_ticks is not None:
        if len(np.array(x_ticks).shape) > 1:
            raise ValueError("x_ticks must be 1D array")
        x = np.linspace(x_ticks[0], x_ticks[-1], len(samples_dicts))
    else:
        x = np.arange(len(samples_dicts))

    plt.rc("text", usetex=True)
    plt.rc("font", size=20, family="serif", weight="bold")
    fig, ax = plt.subplots(figsize=[9, 6])

    nqubits = len(list(samples_dicts[0].keys())[0])
    for i in range(len(strings_qubits)):
        color = cmap(i % 8)
        string_string = "".join(["1" if j in strings_qubits[i] else "0" for j in range(nqubits)])
        string_conf_probabilities = np.array([sd.get(string_string[::-1], 0) / sum(sd.values()) if sum(sd.values()) != 0 else np.nan for sd in samples_dicts], dtype=float)
        plt.plot(x, string_conf_probabilities, "o--", markersize=12, markeredgecolor="black", color=color, label=labels[i] if labels is not None else None)
    
    plt.xlabel(x_label)
    plt.ylabel(r"$P(\mathrm{string})$", labelpad=10)
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    if title:
        plt.title(title, pad=10)
    if labels is not None:
        plt.legend(loc="best", fontsize=12)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()