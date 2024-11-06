from qiskit.primitives.base.base_primitive_job import BasePrimitiveJob
from utils.plotting import convert_jobs_to_site_gauge_matrix
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit_ibm_runtime.runtime_job import RuntimeJob
from z2hex.geometry import is_edge_coords
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def local_observable_plot(site_gauge_matrix_observable_arr, lattice, title=None, scale=1.5, dark_theme=False, filepath=""):
    plt.rc("text", usetex=True)
    plt.rc("font", size=24, family="serif", weight="bold")

    vertices_coords = sorted(list(lattice.vertices.keys()))
    edges_coords = sorted(list(lattice.edges.keys()))
    all_coords = np.array(sorted(vertices_coords + edges_coords))
    is_edge_coords_arr = is_edge_coords(all_coords)

    if dark_theme:
        facecolor = "#373737"
        cmap = plt.get_cmap("inferno")
    else:
        facecolor = "white"
        cmap = truncate_colormap(plt.get_cmap("winter"), 0.4, 1)

    fig, ax = plt.subplots(figsize=[(scale + 0.5)*lattice.max_x, scale*lattice.max_y], facecolor=facecolor)
    ax.set_aspect('equal')
    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    data_cmap_norm = np.max(site_gauge_matrix_observable_arr) - np.min(site_gauge_matrix_observable_arr)
    tdowntransform = mpl.transforms.Affine2D().translate(0, -1*scale)
    trighttransform = mpl.transforms.Affine2D().translate(0.035*scale, 0)
    for i, (y, x) in enumerate(all_coords):
        if is_edge_coords_arr[i]:
            this_edge_endpoints_x = (np.floor(x), np.ceil(x))
            this_edge_endpoints_y = (lattice.max_y - np.floor(y), lattice.max_y - np.ceil(y))
            this_edge_box_x, this_edge_box_y = np.mean([this_edge_endpoints_x, this_edge_endpoints_y], axis=1)
            this_edge_color = cmap(this_norm_data := site_gauge_matrix_observable_arr[i]/data_cmap_norm)
            plt.plot(this_edge_endpoints_x, this_edge_endpoints_y, color="black", linewidth=7*scale)
            plt.plot(this_edge_endpoints_x, this_edge_endpoints_y, color=this_edge_color, linewidth=6*scale)
            plt.scatter(this_edge_box_x, this_edge_box_y, 400*scale, marker=(4, 0, 45), c=this_edge_color, edgecolors="black", zorder=2)
            text_color = "black" if this_norm_data > 0.5 else "white"
            text = plt.text(this_edge_box_x, this_edge_box_y, f"{site_gauge_matrix_observable_arr[i]:.02f}", color=text_color, horizontalalignment="center", verticalalignment="center", fontdict={"size": 4.5*scale, "family":"serif"})
        else:
            vertex_x, vertex_y = x, lattice.max_y - y
            this_node_color = cmap(this_norm_data := site_gauge_matrix_observable_arr[i]/data_cmap_norm)
            plt.scatter(vertex_x, vertex_y, 350*scale, marker="o", c=this_node_color, edgecolors="black", zorder=3)
            text_color = "black" if this_norm_data > 0.5 else "white"
            text = plt.text(vertex_x, vertex_y, f"{site_gauge_matrix_observable_arr[i]:.02f}", c=text_color, horizontalalignment="center", verticalalignment="center", fontdict={"size": 4.5*scale, "family":"serif"})
        text.set_transform(text.get_transform() + tdowntransform + trighttransform)
    plt.axis("off")
    plt.ylim([-0.2, lattice.max_y + 0.2])
    cax = fig.add_axes([0.85, 0.05, 0.02, 0.9])
    cax.tick_params(labelsize=15)
    plt.colorbar(scalar_mappable, cax)
    ax.text(lattice.max_x / 2 + 0.25, lattice.max_y + 0.5, title, horizontalalignment="center", verticalalignment="center", fontdict={"family":"serif", "size": 15*scale})
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.rcdefaults()

def occupation_gif(site_gauge_observable_matrix, lattice, filepath=""):
    if len(lattice) != site_gauge_observable_matrix.shape[1]:
        raise ValueError("site_gauge_observable_matrix and lattice do not have the same number of qubits")