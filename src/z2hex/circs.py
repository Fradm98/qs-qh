# Contributor: César Benito Lamata @cesarBLG

from qiskit.transpiler.passes import Optimize1qGates, SetLayout, ApplyLayout, FullAncillaAllocation, Optimize1qGatesDecomposition, Optimize1qGatesSimpleCommutation
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager, StagedPassManager, Layout
from qiskit.circuit import QuantumCircuit, Parameter
from utils.hexec import backends_objs_to_names
from utils.circs import remove_idle_qwires
from functools import cached_property
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# --------------------------------------
#     HEXAGONAL LATTICE ABSTRACTIONS
# --------------------------------------

# Stores qubit index and coordinates for every vertex (matter qubit)
# Two types of nodes: downwards and upwards (for the schedule)
class vertex:
    def __init__(self, coords, qubit):
        self.coords = coords
        self.qubit = qubit
        self.downwards = (coords[0]+coords[1])%2 == 1

    def __repr__(self):
        return f"<Vertex {self.coords} -> Qb: {self.qubit}>"

# Stores qubit index and coordinates for every edge (gauge qubit)
# Type indicates link direction (not used)
# Color represents a three-colouring of all links (not used)
class edge:
    def __init__(self, coords, qubit):
        self.coords = coords
        self.qubit = qubit
        if int(coords[0]) == coords[0]:
            self.color = [1,0,2][int(coords[1]-0.5)%3]
            self.type = ['x','y'][int(coords[0]+coords[1]-0.5)%2]
        else:
            self.color = [0,1,2][(coords[1]//2+int(coords[0]-0.5)%2+2)%3]
            self.type = 'z'

    def __repr__(self):
        return f"<Edge {self.coords} -> Qb: {self.qubit}>"

# Stores coordinates of all qubits in the heavy-hex lattice
# Additionally, stores edges and vertices in separate lists
class HeavyHexLattice:
    def __init__(self, plaq_width, plaq_height):
        self.plaquettes_width = plaq_width
        self.plaquettes_height = plaq_height
        width=plaq_width*2+1
        height=plaq_height
        edge_coords = []
        vertex_coords = []
        for i in range(height+1):
            for j in range(width+1):
                if i==0 and j == 0:
                    continue
                if i == height:
                    if j == 0 and i%2 != 0:
                        continue
                    if j == width and i%2 == 0:
                        continue
                vertex_coords.append((i,j))
        for i in range(height+1):
            for j in range(width):
                if i==0 and j == 0:
                    continue
                if i == height:
                    if j == 0 and i%2 != 0:
                        continue
                    if j == width-1 and i%2 == 0:
                        continue
                edge_coords.append((i, j+0.5))
        for i in range(height):
            if i%2 == 0:
                for j in range((width+1)//2):
                    edge_coords.append((i+0.5, 2*j+1))
            else:
                for j in range(width//2+1):
                    edge_coords.append((i+0.5, 2*j))
        self.coords = sorted(edge_coords+vertex_coords)
        self.edges = dict()
        self.vertices = dict()
        for (q,c) in enumerate(self.coords):
            if c in edge_coords:
                self.edges[c] = edge(c, q)
            else:
                self.vertices[c] = vertex(c, q)

    def initial_qubit_layout(self, first_qubit=None, backend=None):
        """Returns the mapping from own qubit indices to ibm physical qubits
           Arguments:
            - the output of gen_lattice: coords, vertices and edges
            - the index of the top-left physical qubit to be used
            - the name of the target backend
           Note: for machines other than ibm_fez, qubits are mirrored left to right"""
        backend = backends_objs_to_names(backend)
        reflex = backend != 'ibm_fez' and backend != None
        ibm_qubit_coords = []
        if first_qubit == None:
            if backend == 'ibm_fez':
                first_qubit = 3
            else:
                first_qubit = 0
        if backend == 'ibm_fez':
            for i in range(8):
                for j in range(16):
                    ibm_qubit_coords.append((2*i, j))
            for i in range(4):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+1, 4*j+3))
            for i in range(3):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+3, 4*j+1))
        elif backend == 'ibm_torino':
            for i in range(7):
                for j in range(15):
                    ibm_qubit_coords.append((2*i, j))
            for i in range(4):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+1, 4*j))
            for i in range(3):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+3, 4*j+2))
        elif backend != None: # Eagle r3
            for i in range(7):
                for j in range(15):
                    if i == 0 and j == 14:
                        continue
                    if i == 6 and j == 0:
                        continue
                    ibm_qubit_coords.append((2*i, j))
            for i in range(3):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+1, 4*j))
            for i in range(3):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+3, 4*j+2))
        else:
            ibm_qubit_coords = self.coords
        ibm_qubit_coords = sorted(ibm_qubit_coords)
        ibm_origin = ibm_qubit_coords[first_qubit]
        if reflex:
            own_origin = self.coords[0]
            for c in self.coords:
                if c[0] == 0 and c[1] > own_origin[1]:
                    own_origin = c
        else:
            own_origin = self.coords[0]
        initial_qubit_layout = []
        for c in self.coords:
            offset = (int(2*(c[0]-own_origin[0])), int(2*(c[1]-own_origin[1])))
            if reflex:
                offset = (offset[0],-offset[1])
            ibm_c = (ibm_origin[0]+offset[0],ibm_origin[1]+offset[1])
            initial_qubit_layout.append(ibm_qubit_coords.index(ibm_c))
        return np.array(initial_qubit_layout)

    def __len__(self):
        return len(self.coords)

    @cached_property
    def max_x(self):
        return max([coords[1] for coords in self.coords])
    
    @cached_property
    def max_y(self):
        return max([coords[0] for coords in self.coords])
    
    def plot_lattice(self, scale=1.5, number_qubits=False, initial_qubit = 0):
        plt.rc("font", family="serif")
        vertex_x = np.array([v[1] for v in sorted(list(self.vertices.keys()))], dtype=int)
        vertex_y = np.array([self.max_y - v[0] for v in sorted(list(self.vertices.keys()))], dtype=int)
        edges_endpoints_x = np.array([[np.floor(e[1]), np.ceil(e[1])] for e in sorted(list(self.edges.keys()))], dtype=int)
        edges_endpoints_y = np.array([[self.max_y - np.floor(e[0]), self.max_y - np.ceil(e[0])] for e in sorted(list(self.edges.keys()))], dtype=int)
        edges_boxes_x = np.mean(edges_endpoints_x, axis=1)
        edges_boxes_y = np.mean(edges_endpoints_y, axis=1)
        
        fig, ax = plt.subplots(figsize=[scale*self.max_x, scale*self.max_y])
        ax.set_aspect('equal')
        for exs, eys in zip(edges_endpoints_x, edges_endpoints_y):
            plt.plot(exs, eys, color="black")
        plt.scatter(vertex_x, vertex_y, 350*scale, marker="o", c="white", edgecolors="black", zorder=2)
        plt.scatter(edges_boxes_x, edges_boxes_y, 400*scale, marker=(4, 0, 45), c="white", edgecolors="black", zorder=2)
        if number_qubits:
            ttransform = mpl.transforms.Affine2D().translate(0, -1*scale)
            for i, (y, x) in enumerate(self.coords):
                text = plt.text(x, self.max_y - y, f"{initial_qubit + i}", horizontalalignment="center", verticalalignment="center", fontdict={"size": 5.5*scale})
                text.set_transform(text.get_transform() + ttransform)
        else:
            for i, (x, y) in enumerate(zip(vertex_x, vertex_y)):
                plt.text(x, y, r"$\tau$", horizontalalignment="center", verticalalignment="center", fontdict={"size": 10*scale, "family":"serif"})
            for i, (x, y) in enumerate(zip(edges_boxes_x, edges_boxes_y)):
                plt.text(x, y, r"$\sigma$", horizontalalignment="center", verticalalignment="center", fontdict={"size": 10*scale, "family":"serif"})
        plt.axis("off")
        plt.tight_layout()        

# ------------------------------
#       CIRCUIT PRIMITIVES
# ------------------------------

class StringInitialStateCirc(QuantumCircuit):
    def __init__(self, lattice, string_qubits, x_basis=False):
        super().__init__(len(lattice))
        if x_basis:
            self.h(range(len(lattice)))
            self.z(string_qubits)
        else:
            self.x(string_qubits)
    
class TotalSingleBodyPropagator(QuantumCircuit):
    def __init__(self, lattice, x_basis=False):
        super().__init__(len(lattice))
        t_tau = Parameter("t_τ")
        t_sigma = Parameter("t_σ")
        matter_qubits = [node.qubit for node in lattice.vertices.values()]
        gauge_qubits = [link.qubit for link in lattice.edges.values()]
        if x_basis:
            self.rx(-2*t_tau, matter_qubits)
            self.rx(-2*t_sigma, gauge_qubits)
        else:
            self.rz(-2*t_tau, matter_qubits)
            self.rz(-2*t_sigma, gauge_qubits)

class TotalInteractionPropagator(QuantumCircuit):
    def __init__(self, lattice, x_basis=False):
        super().__init__(len(lattice))
        t_lamb = Parameter("t_λ")
        gauge_qubits = [link.qubit for link in lattice.edges.values()]
        schedule = [(0,-0.5),(-0.5,0),(0,0.5)]
        schedule_donwards = [(0.5,0),(0,-0.5),(0,0.5)]
        for i in range(7):
            for i in range(7):
                if i == 3:
                    if x_basis:
                        self.rz(-2*t_lamb, gauge_qubits)
                    else:
                        self.rx(-2*t_lamb, gauge_qubits)
                else:
                    typ = 6-i if i > 3 else i
                    for v in lattice.vertices.values():
                        this_schedule = schedule_donwards[typ] if v.downwards else schedule[typ]
                        ec = (v.coords[0] + this_schedule[0], v.coords[1] + this_schedule[1])
                        if ec in lattice.edges:
                            if x_basis:
                                self.cx(v.qubit, lattice.edges[ec].qubit)
                            else:
                                self.cx(lattice.edges[ec].qubit, v.qubit)

class TotalInteractionPropagatorDD(QuantumCircuit):
    def __init__(self, lattice, x_basis=False):
        super().__init__(len(lattice))
        t_lamb = Parameter("t_λ")
        t_g = Parameter("t_g")
        gauge_qubits = [link.qubit for link in lattice.edges.values()]
        matter_qubits = [node.qubit for node in lattice.vertices.values()]
        schedule = [(0,-0.5),(-0.5,0),(0,0.5)]
        schedule_downwards = [(0.5,0),(0,-0.5),(0,0.5)]
        for i in range(7):
            for i in range(7):
                if i == 3:
                    if x_basis:
                        self.rz(-2*t_lamb, gauge_qubits)
                        self.rx(-2*t_g, matter_qubits)
                    else:
                        self.rx(-2*t_lamb, gauge_qubits)
                        self.rz(-2*t_g, matter_qubits)
                else:
                    typ = 6-i if i > 3 else i
                    for v in lattice.vertices.values():
                        this_schedule = schedule_donwards[typ] if v.downwards else schedule[typ]
                        ec = (v.coords[0] + this_schedule[0], v.coords[1] + this_schedule[1])
                        if ec in lattice.edges:
                            if x_basis:
                                self.cx(v.qubit, lattice.edges[ec].qubit)
                            else:
                                self.cx(lattice.edges[ec].qubit, v.qubit)

def SecondOrderTrotter(lattice, J, h, lamb, t_total, layers, g=None, x_basis=False, barriers=False):
    t_layer = t_total / layers
    if g is None:
        total_interaction_propagator = TotalInteractionPropagator(lattice, x_basis).decompose()
        total_interaction_propagator.assign_parameters([lamb*t_layer], inplace=True)
    else:
        total_interaction_propagator = TotalInteractionPropagatorDD(lattice, x_basis)
        total_interaction_propagator.assign_parameters([g*t_layer, lamb*t_layer], inplace=True)
    total_single_body_propagator = TotalInteractionPropagatorDD(lattice, x_basis)
    total_single_body_propagator.assign_parameters([h*t_layer/2, t_layer*J/2], inplace=True)
    layer = total_single_body_propagator.compose(total_interaction_propagator).compose(total_single_body_propagator)
    if barriers: layer.barrier()
    return layer.repeat(layers).decompose()

# --------------------------- 
#    CIRCUIT TRANSPILATION
# ---------------------------

def get_erradj_number_of_trotter_layers(times, eplg_absolute, trotter_order, max_layers=100):
    times_matrix = np.repeat(times, max_layers).reshape(len(times), max_layers)
    layers_arr = np.arange(1, max_layers + 1)
    error_trotter = (times_matrix/layers_arr)**(trotter_order + 1)
    error_hardware = 1 - (1-eplg_absolute)**(6*layers_arr)
    error_total = error_trotter + error_hardware
    return np.argmin(error_total, axis=1) + 1

def string_quench_simulation_circuits(lattice, J, h, lamb, string_qubits, final_time, layers, g=None, measure_every_layers=1, x_basis=False, barriers=False):
    initial_state_preparation = StringInitialStateCirc(lattice, string_qubits, x_basis)
    circs_to_return = [initial_state_preparation]
    ncircuits_to_iterate = layers // measure_every_layers
    sqcancel_pm = PassManager([Optimize1qGatesDecomposition()])
    sqopt_pm = StagedPassManager(stages=["optimization"], optimization=sqcancel_pm)
    for i in range(1, ncircuits_to_iterate + 1):
        this_trotter_circuit = SecondOrderTrotter(lattice, J, h, lamb, final_time*i/ncircuits_to_iterate, i*measure_every_layers, g=g, x_basis=x_basis, barriers=barriers)
        this_complete_circuit = initial_state_preparation.compose(this_trotter_circuit)
        this_complete_circuit = sqopt_pm.run(this_complete_circuit)
        circs_to_return.append(this_complete_circuit)
    return circs_to_return

def physical_string_quench_simulation_circuits(lattice, J, h, lamb, string_qubits, final_time, layers, backend, optimization_level, g=None, layout_first_qubit=None, measure_every_layers=1, x_basis=False, barriers=False):
    initial_state_preparation_circ = StringInitialStateCirc(lattice, string_qubits, x_basis)
    logical_trotter_layer_circ = SecondOrderTrotter(lattice, J, h, lamb, final_time/layers, 1, g=g, x_basis=x_basis, barriers=barriers)
    layout = lattice.inital_qubit_layout(layout_first_qubit, backend)
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    physical_trotter_layer_circ = pm.run(logical_trotter_layer_circ)
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=physical_trotter_layer_circ.layout.final_index_layout())
    physical_state_preparation_circ = pm.run(initial_state_preparation_circ)
    circs_to_return = [physical_state_preparation_circ]
    niterations = layers // measure_every_layers
    for i in range(1, niterations + 1):
        this_circuit = physical_state_preparation_circ.compose(physical_trotter_layer_circ.repeat(i*measure_every_layers).decompose())
        this_circuit = remove_idle_qwires(this_circuit)
        if i == 1:
            layout_dict = {layout[i]:this_circuit.qubits[i] for i in range(this_circuit.num_qubits)}
            layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
            sqcancel_pm = PassManager([Optimize1qGates(target=backend.target), Optimize1qGatesDecomposition(target=backend.target), Optimize1qGatesSimpleCommutation(target=backend.target)])
            sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        this_circuit = sqopt_pm.run(this_circuit)
        circs_to_return.append(this_circuit)
    return circs_to_return

def get_erradj_string_quench_simulation_circuits(lattice, J, h, lamb, string_qubits, final_time, steps, backend, optimization_level, eplg_absolute, g=None, layout_first_qubit=None, x_basis=False, barriers=False):
    layout = lattice.initial_qubit_layout(layout_first_qubit, backend)
    t_arr = np.linspace(0, final_time, steps)
    nlayers_arr = get_erradj_number_of_trotter_layers(t_arr, eplg_absolute, trotter_order=2)
    t_perlayer_arr = t_arr / nlayers_arr
    initial_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    state_preparation_logical_circ = StringInitialStateCirc(lattice, string_qubits, x_basis)
    state_preparation_physical_circ = initial_pm.run(state_preparation_logical_circ)
    measured_state_preparation_circuit = QuantumCircuit(len(lattice), len(lattice))
    measured_state_preparation_circuit = measured_state_preparation_circuit.compose(state_preparation_logical_circ)
    state_preparation_physical_circ = initial_pm.run(measured_state_preparation_circuit)
    measured_state_preparation_physical_circ = state_preparation_physical_circ.copy()
    measured_state_preparation_physical_circ.measure(layout[np.argsort(layout)], np.argsort(layout[np.argsort(layout)]))
    circs_to_return = [measured_state_preparation_physical_circ]
    for i in range(steps-1):
        this_trotter_layer_logical_circuit = SecondOrderTrotter(lattice, J, h, lamb, t_perlayer_arr[i+1], 1, g=g, x_basis=x_basis, barriers=barriers)
        this_trotter_layer_physical_circuit = initial_pm.run(this_trotter_layer_logical_circuit)
        state_preparation_physical_circ = initial_pm.run(state_preparation_logical_circ)
        this_time_physical_circuit = state_preparation_physical_circ.compose(this_trotter_layer_physical_circuit.repeat(nlayers_arr[i+1])).decompose()
        this_time_physical_circuit = remove_idle_qwires(this_time_physical_circuit, relabeling=np.arange(len(layout))[np.argsort(layout)])
        layout_dict = {layout[i]:this_time_physical_circuit.qubits[i] for i in range(this_time_physical_circuit.num_qubits)}
        layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
        sqcancel_pm = PassManager([Optimize1qGates(target=backend.target), Optimize1qGatesDecomposition(target=backend.target), Optimize1qGatesSimpleCommutation(target=backend.target)])
        sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        this_time_physical_circuit = sqopt_pm.run(this_time_physical_circuit)
        circs_to_return.append(this_time_physical_circuit)
    return circs_to_return
