# Contributor: César Benito Lamata @cesarBLG

from qiskit.transpiler.passes import Optimize1qGates, SetLayout, ApplyLayout, FullAncillaAllocation, Optimize1qGatesDecomposition, Optimize1qGatesSimpleCommutation, ALAPScheduleAnalysis, PadDelay
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager, StagedPassManager, Layout
from utils.paulis import PauliTensor, PauliTensorSum, commutator
from qiskit.circuit import QuantumCircuit, Parameter
from utils.hexec import backends_objs_to_names
from utils.circs import count_non_idle_qubits
from sympy.physics.paulialgebra import Pauli
from multiprocessing import Pool, cpu_count
from sympy.core.numbers import One
import numpy as np

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
    def __init__(self, lattice, Js, hs, t, x_basis=False):
        super().__init__(len(lattice))
        Js = np.array(Js)
        hs = np.array(hs)
        if len(Js.shape) > 1:
            raise ValueError("The Js parameter must be a vector or scalar")
        elif len(Js.shape) == 1:
            if len(Js) != len(lattice.node_coords):
                raise ValueError("The length of the Js vector must be equal to the number of nodes in the lattice")
        else:
            Js = np.repeat(Js, len(lattice.node_coords))
        if len(hs.shape) > 1:
            raise ValueError("The hs parameter must be a vector or scalar")
        elif len(hs.shape) == 1:
            if len(hs) != len(lattice.edge_coords):
                raise ValueError("The length of the hs vector must be equal to the number of edges in the lattice")
        else:
            hs = np.repeat(hs, len(lattice.edge_coords))

        matter_qubits = [node.qubit for node in lattice.vertices.values()]
        gauge_qubits = [link.qubit for link in lattice.edges.values()]
        for i, qind in enumerate(matter_qubits):
            if x_basis:
                self.rx(-2*Js[i]*t, qind)
            else:
                self.rz(-2*Js[i]*t, qind)
        for i, qind in enumerate(gauge_qubits):
            if x_basis:
                self.rx(-2*hs[i]*t, qind)
            else:
                self.rz(-2*hs[i]*t, qind)

class TotalInteractionPropagator(QuantumCircuit):
    def __init__(self, lattice, lambdas, t, x_basis=False):
        super().__init__(len(lattice))
        lambdas = np.array(lambdas)
        if len(lambdas.shape) > 1:
            raise ValueError("The lambdas parameter must be a vector or scalar")
        elif len(lambdas.shape) == 1:
            if len(lambdas) != len(lattice.edge_coords):
                raise ValueError("The length of the lambdas vector must be equal to the number of edges in the lattice")
        else:
            lambdas = np.repeat(lambdas, len(lattice.edge_coords))
            
        gauge_qubits = [link.qubit for link in lattice.edges.values()]
        schedule = [(0,-0.5),(-0.5,0),(0,0.5)]
        schedule_downwards = [(0.5,0),(0,-0.5),(0,0.5)]
        for i in range(7):
            if i == 3:
                for i, gqb in enumerate(gauge_qubits):
                    if x_basis:
                        self.rz(-2*lambdas[i]*t, gqb)
                    else:
                        self.rx(-2*lambdas[i]*t, gqb)
            else:
                typ = 6-i if i > 3 else i
                for v in lattice.vertices.values():
                    this_schedule = schedule_downwards[typ] if v.downwards else schedule[typ]
                    ec = (v.coords[0] + this_schedule[0], v.coords[1] + this_schedule[1])
                    if ec in lattice.edges:
                        if x_basis:
                            self.cx(v.qubit, lattice.edges[ec].qubit)
                        else:
                            self.cx(lattice.edges[ec].qubit, v.qubit)

class TotalInteractionPropagatorDD(QuantumCircuit):
    def __init__(self, lattice, lambdas, t_g, t, x_basis=False):
        super().__init__(len(lattice))
        lambdas = np.array(lambdas)
        if len(lambdas.shape) > 1:
            raise ValueError("The lambdas parameter must be a vector or scalar")
        elif len(lambdas.shape) == 1:
            if len(lambdas) != len(lattice.edge_coords):
                raise ValueError("The length of the lambdas vector must be equal to the number of edges in the lattice")
        else:
            lambdas = np.repeat(lambdas, len(lattice.edge_coords))
        gauge_qubits = [link.qubit for link in lattice.edges.values()]
        matter_qubits = [node.qubit for node in lattice.vertices.values()]
        t_g = np.array(t_g)
        if t_g.ndim == 0:
            t_g = np.full(len(matter_qubits), t_g)
        elif t_g.ndim > 1:
            raise ValueError(f"t_g should be a scalar or an array of shape ({len(matter_qubits)},)")
        elif t_g.shape[0] != len(matter_qubits):
            raise ValueError(f"t_g should be a scalar or an array of shape ({len(matter_qubits)},)")
        schedule = [(0,-0.5),(-0.5,0),(0,0.5)]
        schedule_downwards = [(0.5,0),(0,-0.5),(0,0.5)]
        # random_times = np.random.uniform(0, t*g, len(matter_qubits))
        for i in range(7):
            if i == 3:
                if x_basis:
                    for i, gqb in enumerate(gauge_qubits):
                        self.rz(-2*t*lambdas[i], gqb)
                    for i, qind in enumerate(matter_qubits):
                        self.rx(-2*t_g[i], qind)
                else:
                    for i, gqb in enumerate(gauge_qubits):
                        self.rx(-2*t*lambdas[i], gqb)
                    for i, qind in enumerate(matter_qubits):
                        self.rz(-2*t_g[i], qind)
            else:
                typ = 6-i if i > 3 else i
                for v in lattice.vertices.values():
                    this_schedule = schedule_downwards[typ] if v.downwards else schedule[typ]
                    ec = (v.coords[0] + this_schedule[0], v.coords[1] + this_schedule[1])
                    if ec in lattice.edges:
                        if x_basis:
                            self.cx(v.qubit, lattice.edges[ec].qubit)
                        else:
                            self.cx(lattice.edges[ec].qubit, v.qubit)

def SecondOrderTrotter(lattice, Js, hs, lambdas, t_total, layers, g=None, x_basis=False, barriers=False):
    t_layer = t_total / layers
    if g is None:
        total_interaction_propagator = TotalInteractionPropagator(lattice, lambdas, t_layer, x_basis).decompose()
    else:
        total_interaction_propagator = TotalInteractionPropagatorDD(lattice, lambdas, g, t_layer, x_basis)
    total_single_body_propagator = TotalSingleBodyPropagator(lattice, Js, hs, t_total/2, x_basis)
    layer = total_single_body_propagator.compose(total_interaction_propagator).compose(total_single_body_propagator)
    if barriers: layer.barrier()
    return layer.repeat(layers).decompose()

# --------------------------- 
#       TROTTER ERROR
# ---------------------------

def hamiltonian_interaction_terms(lattice, lamb):
    edge_coords = lattice.edge_coords
    connected_nodes_coords = [lattice.nodes_connected_to_edge(edge_coord) for edge_coord in edge_coords]
    edge_lowest_node_coords = [nodes_coords[0] for nodes_coords in connected_nodes_coords]
    edge_highest_node_coords = [nodes_coords[1] for nodes_coords in connected_nodes_coords]
    each_interaction_qubits = np.zeros((len(edge_coords), 3), dtype=int)
    each_interaction_qubits[:, 0] = lattice.coords_to_logical_qb(edge_lowest_node_coords)
    each_interaction_qubits[:, 1] = lattice.coords_to_logical_qb(edge_coords)
    each_interaction_qubits[:, 2] = lattice.coords_to_logical_qb(edge_highest_node_coords)
    all_qubits = np.arange(len(lattice))
    tensors_to_sum = [PauliTensor([Pauli(1) if j in this_int_qubits else One for j in all_qubits], coefficient=-lamb) for this_int_qubits in each_interaction_qubits]
    return PauliTensorSum(*tensors_to_sum)

def hamiltonian_diagonal_terms(lattice, J, h):
    matter_qubits = [lattice.coords_to_logical_qb(node_coords) for node_coords in lattice.node_coords]
    all_qubits = np.arange(len(lattice))
    tensors_to_sum = [PauliTensor([Pauli(3) if j == this_qubit else One for j in all_qubits], coefficient=-J if this_qubit in matter_qubits else -h) for this_qubit in all_qubits]
    return PauliTensorSum(*tensors_to_sum)

def trotter_error(t, lattice, J, h, lamb):
    diag_terms = hamiltonian_diagonal_terms(lattice, J, h)
    int_terms = hamiltonian_interaction_terms(lattice, lamb)
    int_int_diag_commutator_norm = commutator(int_terms, commutator(int_terms, diag_terms)).norm_upper_bound()
    diag_diag_int_commutator_norm = commutator(diag_terms, commutator(diag_terms, int_terms)).norm_upper_bound()
    return t**3/12*int_int_diag_commutator_norm + t**3/24*diag_diag_int_commutator_norm

def nlayers_trotter(t, epsilon, lattice, J, h, lamb):
    diag_terms = hamiltonian_diagonal_terms(lattice, J, h)
    int_terms = hamiltonian_interaction_terms(lattice, lamb)
    int_int_diag_commutator_norm = commutator(int_terms, commutator(int_terms, diag_terms)).norm_upper_bound()
    diag_diag_int_commutator_norm = commutator(diag_terms, commutator(diag_terms, int_terms)).norm_upper_bound()
    return t/12/epsilon**(1/3)*int_int_diag_commutator_norm + t/24/epsilon**(1/3)*diag_diag_int_commutator_norm

def trotter_dt(epsilon, lattice, J, h, lamb):
    diag_terms = hamiltonian_diagonal_terms(lattice, J, h)
    int_terms = hamiltonian_interaction_terms(lattice, lamb)
    int_int_diag_commutator_norm = commutator(int_terms, commutator(int_terms, diag_terms)).norm_upper_bound()
    diag_diag_int_commutator_norm = commutator(diag_terms, commutator(diag_terms, int_terms)).norm_upper_bound()
    return (epsilon/(int_int_diag_commutator_norm/12 + diag_diag_int_commutator_norm/24))**(1/3)

def get_erradj_number_of_trotter_layers(times, eplg_absolute, lattice, J, h, lamb, max_layers=50):
    times_matrix = np.repeat(times, max_layers).reshape(len(times), max_layers)
    layers_arr = np.arange(1, max_layers + 1)
    error_trotter = trotter_error(times_matrix/layers_arr, lattice, J, h, lamb)
    error_hardware = 1 - (1-eplg_absolute)**layers_arr
    error_total = error_trotter + error_hardware
    return np.minimum(np.argmin(error_total, axis=1) + 1, max_layers)

# --------------------------- 
#    CIRCUIT TRANSPILATION
# ---------------------------

def string_quench_simulation_circuits(lattice, Js, hs, lambdas, string_qubits, final_time, layers, g=None, measure_every_layers=1, x_basis=False, barriers=False):
    initial_state_preparation = StringInitialStateCirc(lattice, string_qubits, x_basis)
    circs_to_return = [initial_state_preparation]
    ncircuits_to_iterate = layers // measure_every_layers
    sqcancel_pm = PassManager([Optimize1qGatesDecomposition()])
    sqopt_pm = StagedPassManager(stages=["optimization"], optimization=sqcancel_pm)
    for i in range(1, ncircuits_to_iterate + 1):
        this_trotter_circuit = SecondOrderTrotter(lattice, Js, hs, lambdas, final_time*i/ncircuits_to_iterate, i*measure_every_layers, g=g, x_basis=x_basis, barriers=barriers)
        this_complete_circuit = initial_state_preparation.compose(this_trotter_circuit)
        this_complete_circuit = sqopt_pm.run(this_complete_circuit)
        circs_to_return.append(this_complete_circuit)
    return circs_to_return

def get_layouts(lattice, backend, layout_first_or_all=None, forbidden_qubits=[]):
    if "aer_simulator" in backends_objs_to_names(backend):
        return [np.arange(len(lattice))]
    elif layout_first_or_all is None:
        return [lattice.initial_qubit_layout(layout_first_or_all, backend)]
    elif type(layout_first_or_all) == int:
        return [lattice.initial_qubit_layout(layout_first_or_all, backend)]
    elif layout_first_or_all == True:
        all_layouts = lattice.get_all_layouts(backend)
        to_return = []
        for layout in all_layouts:
            for qubit in forbidden_qubits:
                if qubit in layout:
                    break
            else:
                to_return.append(layout)
        return to_return
    
def logical_string_quench_simulation_circuit(lattice, Js, hs, lambdas, string_qubits, dt, nlayers, g_func=None, x_basis=False, barriers=False):
    if callable(g_func):
        g = g_func(lattice, nlayers)
    elif g_func is not None:
        g = g_func
    else:
        g = None
    this_trotter_layers = []
    for k in range(nlayers):
        this_trotter_layers.append(SecondOrderTrotter(lattice, Js, hs, lambdas, dt, 1, g=g[:, k] if g is not None else g, x_basis=x_basis, barriers=barriers))
    this_time_logical_circ = StringInitialStateCirc(lattice, string_qubits, x_basis)
    for trotter_layer in this_trotter_layers:
        this_time_logical_circ = this_time_logical_circ.compose(trotter_layer).decompose()
    if x_basis: this_time_logical_circ.h(range(len(lattice)))
    this_time_logical_circ.measure_all()
    return this_time_logical_circ
    
def string_quench_simulation_circuits_dt(lattice, Js, hs, lambdas, string_qubits, final_time, dt, steps, backend, optimization_level, g_func=None, layouts=None, x_basis=False, force_even=True, barriers=False):
    Js = np.array(Js)
    hs = np.array(hs)
    lambdas = np.array(lambdas)
    if not len(Js.shape) == len(hs.shape) == len(lambdas.shape):
        raise ValueError("The Js, hs and lambdas parameters must have the same shape")
    if len(Js.shape) == 0:
        Js = Js*np.ones((steps, len(lattice.node_coords)))
        hs = hs*np.ones((steps, len(lattice.edge_coords)))
        lambdas = lambdas*np.ones((steps, len(lattice.edge_coords)))
    elif len(Js.shape) == 1:
        Js = np.repeat(Js.reshape(1, -1), steps, axis=0)
        hs = np.repeat(hs.reshape(1, -1), steps, axis=0)
        lambdas = np.repeat(lambdas.reshape(1, -1), steps, axis=0)
    elif len(Js.shape) == 2:
        if Js.shape[0] != steps:
            raise ValueError("The number of rows in the Js, hs and lambdas parameters must be equal to the number of steps")
    else:
        raise ValueError("The parameters Js, hs and lambdas must be scalars, vectors or matrices")
    t_arr = np.linspace(0, final_time, steps+1)
    nlayers_arr = np.round(t_arr / dt).astype(int)[1:]
    nlayers_arr[np.equal(nlayers_arr, 0)] = 1
    if force_even:
        nlayers_arr[~np.equal(nlayers_arr % 2, 0)] += 1
    dt_arr = t_arr[1:] / nlayers_arr
    state_preparation_logical_circ = StringInitialStateCirc(lattice, string_qubits, x_basis)
    measured_state_preparation_logical_circ = state_preparation_logical_circ.copy()
    if x_basis: measured_state_preparation_logical_circ.h(range(len(lattice)))
    measured_state_preparation_logical_circ.measure_all()
    circs_to_return = []
    if layouts is None: layouts = [None]
    for layout in layouts:
        if layout is not None: layout = np.array(layout)
        trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
        this_layout_logical_circs = [measured_state_preparation_logical_circ]
        with Pool(processes=cpu_count()) as pool:
            evolved_circuits = pool.starmap(logical_string_quench_simulation_circuit, [(lattice, Js[i], hs[i], lambdas[i], string_qubits, dt_arr[i], nlayers_arr[i], g_func, x_basis, barriers) for i in range(steps)])
        this_layout_logical_circs += evolved_circuits
        if layout is None:
            this_layout_physical_circs = trotter_pm.run(this_layout_logical_circs[:2])
            first_layout = this_layout_physical_circs[1].layout.final_index_layout()
            trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=first_layout)
            this_layout_physical_circs += trotter_pm.run(this_layout_logical_circs[2:])
        else:
            this_layout_physical_circs = trotter_pm.run(this_layout_logical_circs)
        circs_to_return.append(this_layout_physical_circs)
    return circs_to_return

def logical_odr_compuncomp_circuit(lattice, Js, hs, lambdas, dt, nlayers, g_func=None, string_qubits=None, x_basis=False, barriers=False):
    if callable(g_func):
        g = g_func(lattice, nlayers)
    elif g_func is not None:
        g = g_func
    else:
        g = None
    if string_qubits is not None:
        state_preparation_logical_circ = StringInitialStateCirc(lattice, string_qubits, x_basis)
    else:
        state_preparation_logical_circ = QuantumCircuit(len(lattice))
    this_forwards_trotter_layers = []
    this_backwards_trotter_layers = []
    for k in range(nlayers//2):
        this_forwards_trotter_layers.append(SecondOrderTrotter(lattice, Js, hs, lambdas, dt, 1, g=g[:, k] if g is not None else g, x_basis=x_basis, barriers=barriers))
        this_backwards_trotter_layers.append(this_forwards_trotter_layers[-1].inverse())
    this_time_logical_circ = state_preparation_logical_circ.copy()
    for forwards_trotter_layer in this_forwards_trotter_layers:
        this_time_logical_circ = this_time_logical_circ.compose(forwards_trotter_layer).decompose()
    this_time_logical_circ.barrier()
    for backwards_trotter_layer in reversed(this_backwards_trotter_layers):
        this_time_logical_circ = this_time_logical_circ.compose(backwards_trotter_layer).decompose()
    this_time_logical_circ.measure_all()
    return this_time_logical_circ

def odr_compuncomp_circuits_dt(lattice, Js, hs, lambdas, final_time, dt, steps, backend, optimization_level, g_func=None, string_qubits=None, layouts=None, x_basis=False, barriers=False):
    Js = np.array(Js)
    hs = np.array(hs)
    lambdas = np.array(lambdas)
    if not len(Js.shape) == len(hs.shape) == len(lambdas.shape):
        raise ValueError("The Js, hs and lambdas parameters must have the same shape")
    if len(Js.shape) == 0:
        Js = Js*np.ones((steps, len(lattice.node_coords)))
        hs = hs*np.ones((steps, len(lattice.edge_coords)))
        lambdas = lambdas*np.ones((steps, len(lattice.edge_coords)))
    elif len(Js.shape) == 1:
        Js = np.repeat(Js.reshape(1, -1), steps, axis=0)
        hs = np.repeat(hs.reshape(1, -1), steps, axis=0)
        lambdas = np.repeat(lambdas.reshape(1, -1), steps, axis=0)
    elif len(Js.shape) == 2:
        if Js.shape[0] != steps:
            raise ValueError("The number of rows in the Js, hs and lambdas parameters must be equal to the number of steps")
    else:
        raise ValueError("The parameters Js, hs and lambdas must be scalars, vectors or matrices")
    t_arr = np.linspace(0, final_time, steps+1)
    nlayers_arr = np.round(t_arr / dt).astype(int)[1:]
    nlayers_arr[np.equal(nlayers_arr, 0)] = 1
    nlayers_arr[~np.equal(nlayers_arr % 2, 0)] += 1
    dt_arr = t_arr[1:] / nlayers_arr
    if string_qubits is not None:
        state_preparation_logical_circ = StringInitialStateCirc(lattice, string_qubits, x_basis)
    else:
        state_preparation_logical_circ = QuantumCircuit(len(lattice))
    measured_state_preparation_logical_circ = state_preparation_logical_circ.copy()
    measured_state_preparation_logical_circ.measure_all()
    circs_to_return = []
    if layouts is None: layouts = [None]
    for layout in layouts:
        if layout is not None: layout = np.array(layout)
        trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
        this_layout_logical_circs = [measured_state_preparation_logical_circ]
        with Pool(processes=cpu_count()) as pool:
            evolved_circuits = pool.starmap(logical_odr_compuncomp_circuit, [(lattice, Js[i], hs[i], lambdas[i], dt_arr[i], nlayers_arr[i], g_func, string_qubits, x_basis, barriers) for i in range(steps)])
        this_layout_logical_circs += evolved_circuits
        if layout is None:
            this_layout_physical_circs = trotter_pm.run(this_layout_logical_circs[:2])
            first_layout = this_layout_physical_circs[1].layout.final_index_layout()
            trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=first_layout)
            this_layout_physical_circs += trotter_pm.run(this_layout_logical_circs[2:])
        else:
            this_layout_physical_circs = trotter_pm.run(this_layout_logical_circs)
        circs_to_return.append(this_layout_physical_circs)
    return circs_to_return

def simulation_and_odr_circuits(lattice, Js, hs, lambdas, string_qubits, final_time, dt, steps, backend, optimization_level, g_func=None, layouts=None, x_basis=False, force_even=True, barriers=False):
    seed = np.random.randint(100000)
    np.random.seed(seed)
    simulation_circuits = string_quench_simulation_circuits_dt(lattice, Js, hs, lambdas, string_qubits, final_time, dt, steps, backend, optimization_level, g_func, layouts, x_basis, force_even, barriers)
    np.random.seed(seed)
    if layouts is None:
        layouts = [simulation_circuits[0][1].layout.final_index_layout()]
    odr_circuits = odr_compuncomp_circuits_dt(lattice, Js, hs, lambdas, final_time, dt, steps, backend, optimization_level, g_func, None, layouts, x_basis, barriers)
    np.random.seed()
    return simulation_circuits, odr_circuits