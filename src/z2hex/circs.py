# Contributor: César Benito Lamata @cesarBLG

from qiskit.transpiler.passes import Optimize1qGates, SetLayout, ApplyLayout, FullAncillaAllocation, Optimize1qGatesDecomposition, Optimize1qGatesSimpleCommutation, ALAPScheduleAnalysis, PadDelay
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager, StagedPassManager, Layout
from utils.paulis import PauliTensor, PauliTensorSum, commutator
from qiskit.circuit import QuantumCircuit, Parameter
from utils.hexec import backends_objs_to_names
from sympy.physics.paulialgebra import Pauli
from utils.circs import remove_idle_qwires
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
        schedule_downwards = [(0.5,0),(0,-0.5),(0,0.5)]
        for i in range(7):
            if i == 3:
                if x_basis:
                    self.rz(-2*t_lamb, gauge_qubits)
                else:
                    self.rx(-2*t_lamb, gauge_qubits)
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
    def __init__(self, lattice, t_g, x_basis=False):
        super().__init__(len(lattice))
        t_lamb = Parameter("t_λ")
        gauge_qubits = [link.qubit for link in lattice.edges.values()]
        matter_qubits = [node.qubit for node in lattice.vertices.values()]
        schedule = [(0,-0.5),(-0.5,0),(0,0.5)]
        schedule_downwards = [(0.5,0),(0,-0.5),(0,0.5)]
        random_times = np.random.uniform(0, t_g, len(matter_qubits))
        for i in range(7):
            if i == 3:
                if x_basis:
                    self.rz(-2*t_lamb, gauge_qubits)
                    for i, qind in enumerate(matter_qubits):
                        self.rx(-2*random_times[i], qind)
                else:
                    self.rx(-2*t_lamb, gauge_qubits)
                    for i, qind in enumerate(matter_qubits):
                        self.rz(-2*random_times[i], qind)
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

def SecondOrderTrotter(lattice, J, h, lamb, t_total, layers, g=None, x_basis=False, barriers=False):
    t_layer = t_total / layers
    if g is None:
        total_interaction_propagator = TotalInteractionPropagator(lattice, x_basis).decompose()
        total_interaction_propagator.assign_parameters([lamb*t_layer], inplace=True)
    else:
        total_interaction_propagator = TotalInteractionPropagatorDD(lattice, g*t_layer, x_basis)
        total_interaction_propagator.assign_parameters([lamb*t_layer], inplace=True)
    total_single_body_propagator = TotalSingleBodyPropagator(lattice, x_basis)
    total_single_body_propagator.assign_parameters([h*t_layer/2, t_layer*J/2], inplace=True)
    layer = total_single_body_propagator.compose(total_interaction_propagator).compose(total_single_body_propagator)
    if barriers: layer.barrier()
    return layer.repeat(layers).decompose()

# --------------------------- 
#    CIRCUIT TRANSPILATION
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
    all_qubits = np.arange(len(lattice))
    tensors_to_sum = [PauliTensor([Pauli(3) if j == this_qubit else One for j in all_qubits], coefficient=-J if this_qubit%2==0 else -h) for this_qubit in all_qubits]
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

def string_quench_simulation_circuits_dt(lattice, J, h, lamb, string_qubits, final_time, dt, steps, backend, optimization_level, g=None, layout_first_qubit=None, x_basis=False, force_even=True, barriers=False):
    backend_name = backends_objs_to_names(backend)
    if "aer_simulator" in backend_name:
        layout = np.arange(len(lattice))
    else:
        layout = lattice.initial_qubit_layout(layout_first_qubit, backend)
    t_arr = np.linspace(0, final_time, steps+1)
    nlayers_arr = (t_arr // dt).astype(int)
    nlayers_arr[nlayers_arr == 0] = 1
    if force_even:
        nlayers_arr[np.equal(nlayers_arr % 2, 0)] += 1
    dt_arr = t_arr / nlayers_arr
    state_preparation_logical_circ = StringInitialStateCirc(lattice, string_qubits, x_basis)
    trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    state_preparation_physical_circ = trotter_pm.run(state_preparation_logical_circ)
    try:
        if (fractional_gates := backend.options.use_fractional_gates):
            sched_pm = PassManager([ALAPScheduleAnalysis(backend.instruction_durations), PadDelay(backend.instruction_durations)])
    except AttributeError:
        fractional_gates = False
    measured_state_preparation_circuit = QuantumCircuit(len(lattice), len(lattice))
    measured_state_preparation_circuit = measured_state_preparation_circuit.compose(state_preparation_logical_circ)
    state_preparation_physical_circ = trotter_pm.run(measured_state_preparation_circuit)
    measured_state_preparation_physical_circ = state_preparation_physical_circ.copy()
    measured_state_preparation_physical_circ.measure(layout[np.argsort(layout)], np.argsort(layout[np.argsort(layout)]))
    if fractional_gates:
        measured_state_preparation_physical_circ = sched_pm.run(measured_state_preparation_physical_circ)
    circs_to_return = [measured_state_preparation_physical_circ]
    sqcancel_pm = PassManager([Optimize1qGates(target=backend.target), Optimize1qGatesDecomposition(target=backend.target), Optimize1qGatesSimpleCommutation(target=backend.target)])
    for i in range(steps):
        this_trotter_layer_logical_circuit = SecondOrderTrotter(lattice, J, h, lamb, dt_arr[i+1], 1, g=g, x_basis=x_basis, barriers=barriers)
        this_trotter_layer_physical_circuit = trotter_pm.run(this_trotter_layer_logical_circuit)
        state_preparation_physical_circ = trotter_pm.run(state_preparation_logical_circ)
        this_time_physical_circuit = state_preparation_physical_circ.compose(this_trotter_layer_physical_circuit.repeat(nlayers_arr[i+1])).decompose()
        this_time_physical_circuit = remove_idle_qwires(this_time_physical_circuit, relabeling=np.arange(len(layout))[np.argsort(layout)])
        layout_dict = {layout[i]:this_time_physical_circuit.qubits[i] for i in range(this_time_physical_circuit.num_qubits)}
        layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
        sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        this_time_physical_circuit = sqopt_pm.run(this_time_physical_circuit)
        circs_to_return.append(this_time_physical_circuit)
    return circs_to_return

def all_layouts_string_quench_simulation_circuits_dt(lattice, J, h, lamb, string_qubits, final_time, dt, steps, backend, optimization_level, g=None, forbidden_qubits=[], x_basis=False, force_even=True, barriers=False):
    all_layouts = lattice.all_qubit_layouts(backend)
    to_construct_layouts = []
    for layout in all_layouts:
        for qubit in forbidden_qubits:
            if qubit in layout:
                break
        else:
            to_construct_layouts.append(layout)
    t_arr = np.linspace(0, final_time, steps+1)
    nlayers_arr = (t_arr // dt).astype(int)
    nlayers_arr[nlayers_arr == 0] = 1
    if force_even:
        nlayers_arr[np.equal(nlayers_arr % 2, 0)] += 1
    dt_arr = t_arr / nlayers_arr
    state_preparation_logical_circ = StringInitialStateCirc(lattice, string_qubits, x_basis)
    circs_to_return = []
    for layout in to_construct_layouts:
        trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
        state_preparation_physical_circ = trotter_pm.run(state_preparation_logical_circ)
        try:
            if (fractional_gates := backend.options.use_fractional_gates):
                sched_pm = PassManager([ALAPScheduleAnalysis(backend.instruction_durations), PadDelay(backend.instruction_durations)])
        except AttributeError:
            fractional_gates = False
        measured_state_preparation_circuit = QuantumCircuit(len(lattice), len(lattice))
        measured_state_preparation_circuit = measured_state_preparation_circuit.compose(state_preparation_logical_circ)
        state_preparation_physical_circ = trotter_pm.run(measured_state_preparation_circuit)
        measured_state_preparation_physical_circ = state_preparation_physical_circ.copy()
        measured_state_preparation_physical_circ.measure(layout[np.argsort(layout)], np.argsort(layout[np.argsort(layout)]))
        if fractional_gates:
            measured_state_preparation_physical_circ = sched_pm.run(measured_state_preparation_physical_circ)
        this_circs_to_return = [measured_state_preparation_physical_circ]
        sqcancel_pm = PassManager([Optimize1qGates(target=backend.target), Optimize1qGatesDecomposition(target=backend.target), Optimize1qGatesSimpleCommutation(target=backend.target)])
        for i in range(steps):
            this_trotter_layer_logical_circuit = SecondOrderTrotter(lattice, J, h, lamb, dt_arr[i+1], 1, g=g, x_basis=x_basis, barriers=barriers)
            this_trotter_layer_physical_circuit = trotter_pm.run(this_trotter_layer_logical_circuit)
            state_preparation_physical_circ = trotter_pm.run(state_preparation_logical_circ)
            this_time_physical_circuit = state_preparation_physical_circ.compose(this_trotter_layer_physical_circuit.repeat(nlayers_arr[i+1])).decompose()
            this_time_physical_circuit = remove_idle_qwires(this_time_physical_circuit, relabeling=np.arange(len(layout))[np.argsort(layout)])
            layout_dict = {layout[i]:this_time_physical_circuit.qubits[i] for i in range(this_time_physical_circuit.num_qubits)}
            layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
            sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
            this_time_physical_circuit = sqopt_pm.run(this_time_physical_circuit)
            this_circs_to_return.append(this_time_physical_circuit)
        circs_to_return.append(this_circs_to_return)
    return circs_to_return

def erradj_string_quench_simulation_circuits_eplg(lattice, J, h, lamb, string_qubits, final_time, steps, backend, optimization_level, eplg_absolute, g=None, layout_first_qubit=None, x_basis=False, force_even=True, max_layers=100, adjust_depth=False, barriers=False):
    backend_name = backends_objs_to_names(backend)
    if "aer_simulator" in backend_name:
        layout = np.arange(len(lattice))
    else:
        layout = lattice.initial_qubit_layout(layout_first_qubit, backend)
    t_arr = np.linspace(0, final_time, steps+1)
    # nlayers_arr = get_erradj_number_of_trotter_layers(t_arr, eplg_absolute, trotter_order=2, max_layers=max_layers)
    if adjust_depth:
        last_circ_nlayers_trotter = nlayers_trotter(final_time, eplg_absolute, lattice, J, h, lamb)
        depth_constant = max_layers / last_circ_nlayers_trotter if max_layers < last_circ_nlayers_trotter else 1
        nlayers_arr = np.ceil(depth_constant*nlayers_trotter(t_arr, eplg_absolute, lattice, J, h, lamb)).astype(int)
        nlayers_arr[np.equal(nlayers_arr, 0)] += 1
    else:
        nlayers_arr = get_erradj_number_of_trotter_layers(t_arr, eplg_absolute, lattice, J, h, lamb, max_layers)
    if force_even:
        odd_layers = ~np.equal(nlayers_arr % 2, 0)
        nlayers_arr[odd_layers] = nlayers_arr[odd_layers] + 1
    t_perlayer_arr = t_arr / nlayers_arr
    state_preparation_logical_circ = StringInitialStateCirc(lattice, string_qubits, x_basis)
    trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    state_preparation_physical_circ = trotter_pm.run(state_preparation_logical_circ)
    try:
        if (fractional_gates := backend.options.use_fractional_gates):
            sched_pm = PassManager([ALAPScheduleAnalysis(backend.instruction_durations), PadDelay(backend.instruction_durations)])
    except AttributeError:
        fractional_gates = False
    measured_state_preparation_circuit = QuantumCircuit(len(lattice), len(lattice))
    measured_state_preparation_circuit = measured_state_preparation_circuit.compose(state_preparation_logical_circ)
    state_preparation_physical_circ = trotter_pm.run(measured_state_preparation_circuit)
    measured_state_preparation_physical_circ = state_preparation_physical_circ.copy()
    measured_state_preparation_physical_circ.measure(layout[np.argsort(layout)], np.argsort(layout[np.argsort(layout)]))
    if fractional_gates:
        measured_state_preparation_physical_circ = sched_pm.run(measured_state_preparation_physical_circ)
    circs_to_return = [measured_state_preparation_physical_circ]
    sqcancel_pm = PassManager([Optimize1qGates(target=backend.target), Optimize1qGatesDecomposition(target=backend.target), Optimize1qGatesSimpleCommutation(target=backend.target)])
    for i in range(steps):
        this_trotter_layer_logical_circuit = SecondOrderTrotter(lattice, J, h, lamb, t_perlayer_arr[i+1], 1, g=g, x_basis=x_basis, barriers=barriers)
        this_trotter_layer_physical_circuit = trotter_pm.run(this_trotter_layer_logical_circuit)
        state_preparation_physical_circ = trotter_pm.run(state_preparation_logical_circ)
        this_time_physical_circuit = state_preparation_physical_circ.compose(this_trotter_layer_physical_circuit.repeat(nlayers_arr[i+1])).decompose()
        this_time_physical_circuit = remove_idle_qwires(this_time_physical_circuit, relabeling=np.arange(len(layout))[np.argsort(layout)])
        layout_dict = {layout[i]:this_time_physical_circuit.qubits[i] for i in range(this_time_physical_circuit.num_qubits)}
        layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
        sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        this_time_physical_circuit = sqopt_pm.run(this_time_physical_circuit)
        circs_to_return.append(this_time_physical_circuit)
    return circs_to_return

def odr_compuncomp_circuits_eplg(lattice, J, h, lamb, final_time, steps, backend, eplg_absolute, g=None, string_qubits=None, layout_first_qubit=None, x_basis=False, max_layers=50, adjust_depth=False, barriers=False):
    backend_name = backends_objs_to_names(backend)
    if "aer_simulator" in backend_name:
        layout = np.arange(len(lattice))
    else:
        layout = lattice.initial_qubit_layout(layout_first_qubit, backend)
    t_arr = np.linspace(0, final_time, steps+1)
    # nlayers_arr = get_erradj_number_of_trotter_layers(t_arr, eplg_absolute, trotter_order=2, max_layers=max_layers)
    if adjust_depth:
        last_circ_nlayers_trotter = nlayers_trotter(final_time, eplg_absolute, lattice, J, h, lamb)
        depth_constant = max_layers / last_circ_nlayers_trotter if max_layers < last_circ_nlayers_trotter else 1
        nlayers_arr = np.ceil(depth_constant*nlayers_trotter(t_arr, eplg_absolute, lattice, J, h, lamb)).astype(int)
        nlayers_arr[np.equal(nlayers_arr, 0)] += 1
    else:
        nlayers_arr = get_erradj_number_of_trotter_layers(t_arr, eplg_absolute, lattice, J, h, lamb, max_layers)
    odd_layers = ~np.equal(nlayers_arr % 2, 0)
    nlayers_arr[odd_layers] = nlayers_arr[odd_layers] + 1
    t_perlayer_arr = t_arr / nlayers_arr
    trotter_pm = generate_preset_pass_manager(optimization_level=0, backend=backend, initial_layout=layout)
    if string_qubits is None:
        first_circuit = QuantumCircuit(len(lattice))
        first_circuit.measure_all()
        first_physical_circuit = trotter_pm.run(first_circuit)
    else:
        first_circuit = StringInitialStateCirc(lattice, string_qubits, x_basis)
        first_physical_circuit = QuantumCircuit(len(lattice), len(lattice))
        first_physical_circuit.compose(first_physical_circuit)
        first_physical_circuit = trotter_pm.run(first_physical_circuit)
        final_index_layout = first_physical_circuit.layout.final_index_layout()
        first_physical_circuit.measure(np.array(final_index_layout)[np.argsort(final_index_layout)], np.argsort(np.array(final_index_layout)[np.argsort(final_index_layout)]))
    circs_to_return = [first_physical_circuit]
    sqcancel_pm = PassManager([Optimize1qGates(target=backend.target), Optimize1qGatesDecomposition(target=backend.target), Optimize1qGatesSimpleCommutation(target=backend.target)])
    for i in range(steps):
        forwards_trotter_layer = SecondOrderTrotter(lattice, J, h, lamb, t_perlayer_arr[i+1], 1, g, x_basis, barriers)
        backwards_trotter_layer = SecondOrderTrotter(lattice, J, h, lamb, -t_perlayer_arr[i+1], 1, g, x_basis, barriers)
        forwards_physical_trotter_layer = trotter_pm.run(forwards_trotter_layer)
        backwards_physical_trotter_layer = trotter_pm.run(backwards_trotter_layer)
        this_time_circuit = forwards_physical_trotter_layer.repeat(nlayers_arr[i+1]//2)
        this_time_circuit = this_time_circuit.compose(backwards_physical_trotter_layer.repeat(nlayers_arr[i+1]//2)).decompose()
        this_time_circuit = remove_idle_qwires(this_time_circuit, relabeling=np.arange(len(layout))[np.argsort(layout)])
        layout_dict = {layout[i]:this_time_circuit.qubits[i] for i in range(this_time_circuit.num_qubits)}
        layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
        sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        this_time_circuit = sqopt_pm.run(this_time_circuit)
        circs_to_return.append(this_time_circuit)
    return circs_to_return

def odr_compuncomp_circuits_dt(lattice, J, h, lamb, final_time, dt, steps, backend, g=None, string_qubits=None, layout_first_qubit=None, x_basis=False, barriers=False):
    backend_name = backends_objs_to_names(backend)
    if "aer_simulator" in backend_name:
        layout = np.arange(len(lattice))
    else:
        layout = lattice.initial_qubit_layout(layout_first_qubit, backend)
    t_arr = np.linspace(0, final_time, steps+1)
    nlayers_arr = (t_arr // dt).astype(int)
    nlayers_arr[nlayers_arr == 0] = 1
    odd_layers = ~np.equal(nlayers_arr % 2, 0)
    nlayers_arr[odd_layers] = nlayers_arr[odd_layers] + 1
    dt_arr = t_arr / nlayers_arr
    trotter_pm = generate_preset_pass_manager(optimization_level=0, backend=backend, initial_layout=layout)
    if string_qubits is None:
        first_circuit = QuantumCircuit(len(lattice))
        first_circuit.measure_all()
        first_physical_circuit = trotter_pm.run(first_circuit)
    else:
        first_circuit = StringInitialStateCirc(lattice, string_qubits, x_basis)
        first_physical_circuit = QuantumCircuit(len(lattice), len(lattice))
        first_physical_circuit.compose(first_physical_circuit)
        first_physical_circuit = trotter_pm.run(first_physical_circuit)
        final_index_layout = first_physical_circuit.layout.final_index_layout()
        first_physical_circuit.measure(np.array(final_index_layout)[np.argsort(final_index_layout)], np.argsort(np.array(final_index_layout)[np.argsort(final_index_layout)]))
    circs_to_return = [first_physical_circuit]
    sqcancel_pm = PassManager([Optimize1qGates(target=backend.target), Optimize1qGatesDecomposition(target=backend.target), Optimize1qGatesSimpleCommutation(target=backend.target)])
    for i in range(steps):
        forwards_trotter_layer = SecondOrderTrotter(lattice, J, h, lamb, dt_arr[i+1], 1, g, x_basis, barriers)
        backwards_trotter_layer = SecondOrderTrotter(lattice, J, h, lamb, -dt_arr[i+1], 1, g, x_basis, barriers)
        forwards_physical_trotter_layer = trotter_pm.run(forwards_trotter_layer)
        backwards_physical_trotter_layer = trotter_pm.run(backwards_trotter_layer)
        this_time_circuit = forwards_physical_trotter_layer.repeat(nlayers_arr[i+1]//2)
        this_time_circuit = this_time_circuit.compose(backwards_physical_trotter_layer.repeat(nlayers_arr[i+1]//2)).decompose()
        this_time_circuit = remove_idle_qwires(this_time_circuit, relabeling=np.arange(len(layout))[np.argsort(layout)])
        layout_dict = {layout[i]:this_time_circuit.qubits[i] for i in range(this_time_circuit.num_qubits)}
        layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
        sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        this_time_circuit = sqopt_pm.run(this_time_circuit)
        circs_to_return.append(this_time_circuit)
    return circs_to_return