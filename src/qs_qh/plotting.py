import matplotlib.pyplot as plt
import numpy as np

# default parameters of the plot layout
plt.rcParams["text.usetex"] = True  # use latex
plt.rcParams["font.size"] = 13
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.constrained_layout.use"] = True

# ---------------------------------------------------------------------------------------
# Create Sequential Colors
# ---------------------------------------------------------------------------------------
def create_sequential_colors(num_colors, colormap_name: str='viridis'):
    """
    create_sequential_colors

    This function creates a sequence of colors extracted from a specified colormap.

    num_colors: int - number of colors we want to extract
    colormap_name: string - colormap name we want to use

    """
    colormap = plt.cm.get_cmap(colormap_name)
    colormap_values = np.linspace(0, 1, num_colors)
    colors = [colormap(value) for value in colormap_values]
    return colors


def plot_colormap(data, qubits, depths, zne_extrapolator, mem):
    plt.figure(figsize=(8, 6), tight_layout=True)
    plt.matshow(data, origin='lower', cmap='Greys', interpolation='nearest')

    # Add title and labels
    plt.title('$\langle \hat{Z}\\rangle$ error | '+f'ZNE: {zne_extrapolator}, MEM: {mem}')
    plt.xlabel('depth (2-qubit gates)')
    plt.xticks(ticks=np.linspace(0,len(depths)-1,len(depths)), labels=depths)
    plt.ylabel('qubits')
    plt.yticks(ticks=np.linspace(0,len(qubits)-1,len(qubits)), labels=qubits)

    # Show color bar to indicate the values
    plt.colorbar()

    # Add text annotations
    color = "deepskyblue"
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color=color)

    plt.show()

def plot_results(data, qubits, depths, zne_extrapolator, mem):
    color = create_sequential_colors(len(depths), "viridis")
    for i, error in enumerate(data):
        plt.plot(depths, error, color=color[i], label=f"qubits: {qubits[i]}")

    plt.title('$\langle \hat{Z}\\rangle$ error | '+f'ZNE: {zne_extrapolator}, MEM: {mem}')
    plt.xlabel('depth (2-qubit gates)')
    plt.xticks(ticks=np.linspace(depths[0],depths[-1], len(depths)), labels=depths)
    plt.ylabel('$\langle \hat{Z}\\rangle$ mean error')
    plt.legend()