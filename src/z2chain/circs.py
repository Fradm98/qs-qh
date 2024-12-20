from qiskit.transpiler.passes import Optimize1qGates, SetLayout, ApplyLayout, FullAncillaAllocation, Optimize1qGatesDecomposition, Optimize1qGatesSimpleCommutation
from qiskit_ibm_runtime.transpiler.passes.scheduling import ALAPScheduleAnalysis, PadDelay
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager, StagedPassManager, Layout
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.converters import circuit_to_instruction
from utils.circs import remove_idle_qwires
import numpy as np

class LocalInteractionPropagator(QuantumCircuit):
    def __init__(self, x_basis=False):
        super().__init__(3)
        t = Parameter("t")
        if x_basis:
            self.cx(0, 1)
            self.cx(2, 1)
            self.rx(t, 1)
            self.cx(2, 1)
            self.cx(0, 1)
        else:
            self.cx(1, 0)
            self.cx(1, 2)
            self.rx(t, 1)
            self.cx(1, 2)
            self.cx(1, 0)

class LocalInteractionPropagatorFirst(QuantumCircuit):
    def __init__(self, x_basis=False):
        super().__init__(2)
        if x_basis:
            self.cx(0, 1)
        else:
            self.cx(1, 0)
        
class LocalInteractionPropagatorSecond(QuantumCircuit):
    def __init__(self, x_basis=False):
        super().__init__(2)
        if x_basis:
            self.cx(1, 0)
        else:
            self.cx(0, 1)

class TotalSingleBodyPropagator(QuantumCircuit):
    def __init__(self, chain_length, x_basis=False):
        nqubits = 2*chain_length - 1
        super().__init__(nqubits)
        t_tau = Parameter("t_τ")
        t_sigma = Parameter("t_σ")
        if x_basis:
            self.rx(-2*t_tau, np.arange(0, self.num_qubits, 2))
            self.rx(-2*t_sigma, np.arange(1, self.num_qubits, 2))
        else:
            self.rz(-2*t_tau, np.arange(0, self.num_qubits, 2))
            self.rz(-2*t_sigma, np.arange(1, self.num_qubits, 2))

class TotalInteractionPropagator(QuantumCircuit):
    def __init__(self, chain_length, x_basis=False):
        nqubits = 2*chain_length - 1
        first_cnots_trgt_qubits = np.arange(0, nqubits-1, 2)
        first_cnots_ctrl_qubits = np.arange(1, nqubits, 2)
        second_cnots_trgt_qubits = np.arange(2, nqubits, 2)
        t_lamb = Parameter("t_lamb")
        super().__init__(nqubits)
        if x_basis:
            self.cx(first_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.cx(second_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.rz(-2*t_lamb, np.arange(1, nqubits, 2))
            self.cx(second_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.cx(first_cnots_trgt_qubits, first_cnots_ctrl_qubits)
        else:
            self.cx(first_cnots_ctrl_qubits, first_cnots_trgt_qubits)
            self.cx(first_cnots_ctrl_qubits, second_cnots_trgt_qubits)
            self.rx(-2*t_lamb, np.arange(1, nqubits, 2))
            self.cx(first_cnots_ctrl_qubits, second_cnots_trgt_qubits)
            self.cx(first_cnots_ctrl_qubits, first_cnots_trgt_qubits)

class TotalInteractionPropagatorDD(QuantumCircuit):
    def __init__(self, chain_length, t_g, std_tg_factor=0.5, x_basis=False):
        nqubits = 2*chain_length - 1
        first_cnots_trgt_qubits = np.arange(0, nqubits-1, 2)
        first_cnots_ctrl_qubits = np.arange(1, nqubits, 2)
        second_cnots_trgt_qubits = np.arange(2, nqubits, 2)
        t_lamb = Parameter("t_lamb")
        # random_times = np.random.normal(t_g, std_tg_factor*np.abs(t_g), chain_length)
        random_times = np.random.uniform(0, t_g, chain_length)
        super().__init__(nqubits)
        if x_basis:
            self.cx(first_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.cx(second_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            for i, qind in enumerate(range(0, nqubits, 2)):
                self.rx(-2*random_times[i], qind)
            self.rz(-2*t_lamb, np.arange(1, nqubits, 2))
            self.cx(second_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.cx(first_cnots_trgt_qubits, first_cnots_ctrl_qubits)
        else:
            self.cx(first_cnots_ctrl_qubits, first_cnots_trgt_qubits)
            self.cx(first_cnots_ctrl_qubits, second_cnots_trgt_qubits)
            for i, qind in enumerate(range(0, nqubits, 2)):
                self.rz(-2*random_times[i], qind)
            self.rx(-2*t_lamb, np.arange(1, nqubits, 2))
            self.cx(first_cnots_ctrl_qubits, second_cnots_trgt_qubits)
            self.cx(first_cnots_ctrl_qubits, first_cnots_trgt_qubits)

class TotalInteractionPropagatorOld(QuantumCircuit):
    def __init__(self, chain_length, x_basis=False):
        nqubits = 2*chain_length - 1
        qubit_list = np.arange(nqubits)
        int_qb_inds_first = qubit_list[np.less(qubit_list % 4, 3)][:-1 if chain_length % 2 else None]
        int_qb_inds_second = qubit_list[2::][np.less((qubit_list[2::] - 2) % 4, 3)][:None if chain_length % 2 else -1]
        super().__init__(nqubits)
        local_prop = LocalInteractionPropagator(x_basis)
        local_prop_instruction = circuit_to_instruction(local_prop)
        for qubits in int_qb_inds_first.reshape((len(int_qb_inds_first)//3, 3)):
            self.append(local_prop_instruction, list(qubits))

        for qubits in int_qb_inds_second.reshape((len(int_qb_inds_second)//3, 3)):
            self.append(local_prop_instruction, list(qubits))

class particle_pair_initial_state(QuantumCircuit):
    def __init__(self, chain_length, left_particle_position, particle_pair_length=1, x_basis=False):
        nqubits = 2*chain_length - 1
        super().__init__(nqubits)
        if particle_pair_length > 0:
            if x_basis:
                self.h(range(nqubits))
                self.z(range(2*left_particle_position, 2*(left_particle_position + particle_pair_length) + 1))
            else:
                self.x(range(2*left_particle_position, 2*(left_particle_position + particle_pair_length) + 1))

def FirstOrderTrotter(chain_length, J, h, lamb, t_total, layers, sqrot_first=False, x_basis=False, barriers=False):
    t_layer = t_total/layers
    total_interaction_propagator = TotalInteractionPropagator(chain_length, x_basis).decompose()
    total_interaction_propagator.assign_parameters([lamb*t_layer], inplace=True)
    total_single_body_propagator = TotalSingleBodyPropagator(chain_length, x_basis)
    total_single_body_propagator.assign_parameters([h*t_layer, t_layer*J], inplace=True)
    if sqrot_first:
        layer = total_single_body_propagator.compose(total_interaction_propagator)
    else:
        layer = total_interaction_propagator.compose(total_single_body_propagator)
    if barriers:
        layer.barrier()
    return layer.repeat(layers).decompose()

def SecondOrderTrotter(chain_length, J, h, lamb, t_total, layers, g=None, x_basis=False, barriers=False):
    t_layer = t_total/layers
    if g is None:
        total_interaction_propagator = TotalInteractionPropagator(chain_length, x_basis).decompose()
    else:
        total_interaction_propagator = TotalInteractionPropagatorDD(chain_length, t_layer*g, x_basis=x_basis)
    total_interaction_propagator.assign_parameters([lamb*t_layer], inplace=True)
    total_single_body_propagator = TotalSingleBodyPropagator(chain_length, x_basis)
    total_single_body_propagator.assign_parameters([h*t_layer/2, t_layer*J/2], inplace=True)
    layer = total_single_body_propagator.compose(total_interaction_propagator).compose(total_single_body_propagator)
    if barriers: layer.barrier()
    return layer.repeat(layers).decompose()

def SecondOrderTrotterOld(chain_length, J, h, lamb, t_total, layers, x_basis=False, barriers=False):
    t_layer = t_total/layers
    # total_interaction_propagator = TotalInteractionPropagatorOld(chain_length, x_basis)
    total_interaction_propagator = TotalInteractionPropagatorOld(chain_length, x_basis).decompose()
    total_interaction_propagator.assign_parameters([lamb*t_layer], inplace=True)
    total_single_body_propagator = TotalSingleBodyPropagator(chain_length, x_basis)
    total_single_body_propagator.assign_parameters([h*t_layer/2, t_layer*J/2], inplace=True)
    layer = total_single_body_propagator.compose(total_interaction_propagator).compose(total_single_body_propagator)
    if barriers: layer.barrier()
    return layer.repeat(layers).decompose()

def particle_pair_quench_simulation_circuits(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, layers, g=None, measure_every_layers=1, x_basis=False, barriers=False):
    initial_state_preparation = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    circs_to_return = [initial_state_preparation]
    ncircuits_to_iterate = layers // measure_every_layers
    sqcancel_pm = PassManager([Optimize1qGatesDecomposition()])
    sqopt_pm = StagedPassManager(stages=["optimization"], optimization=sqcancel_pm)
    for i in range(1, ncircuits_to_iterate + 1):
        this_trotter_circuit = SecondOrderTrotter(chain_length, J, h, lamb, final_time*i/ncircuits_to_iterate, i*measure_every_layers, g=g, x_basis=x_basis, barriers=barriers)
        this_complete_circuit = initial_state_preparation.compose(this_trotter_circuit)
        this_complete_circuit = sqopt_pm.run(this_complete_circuit)
        circs_to_return.append(this_complete_circuit)
    return circs_to_return

def physical_particle_pair_quench_simulation_circuits(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, layers, backend, optimization_level, g=None, layout=None, measure_every_layers=1, x_basis=False, barriers=False):
    initial_state_preparation_circ = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    logical_trotter_layer_circ = SecondOrderTrotter(chain_length, J, h, lamb, final_time/layers, 1, g=g, x_basis=x_basis, barriers=barriers)
    # display(logical_trotter_layer_circ.draw(output="mpl", idle_wires=False, fold=-1))
    layout = layout[:logical_trotter_layer_circ.num_qubits] if layout is not None else None
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    physical_trotter_layer_circ = pm.run(logical_trotter_layer_circ)
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=physical_trotter_layer_circ.layout.final_index_layout())
    physical_state_preparation_circuit = pm.run(initial_state_preparation_circ)
    circs_to_return = [physical_state_preparation_circuit]
    niterations = layers // measure_every_layers
    for i in range(1, niterations + 1):
        this_circuit = physical_state_preparation_circuit.compose(physical_trotter_layer_circ.repeat(i*measure_every_layers).decompose())
        this_circuit = remove_idle_qwires(this_circuit)
        if i == 1:
            if layout is not None:
                layout_dict = {layout[i]:this_circuit.qubits[i] for i in range(this_circuit.num_qubits)}
            else:
                layout_dict = {physical_trotter_layer_circ.layout.final_index_layout()[i]:this_circuit.qubits[i] for i in range(this_circuit.num_qubits)}
            layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
            sqcancel_pm = PassManager([Optimize1qGates(target=backend.target), Optimize1qGatesDecomposition(target=backend.target), Optimize1qGatesSimpleCommutation(target=backend.target)])
            sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        this_circuit = sqopt_pm.run(this_circuit)
        circs_to_return.append(this_circuit)
    return circs_to_return

def physical_particle_pair_quench_simulation_circuits_old(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, layers, backend, optimization_level, layout=None, measure_every_layers=1, x_basis=False, barriers=False):
    initial_state_preparation_circ = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    logical_trotter_layer_circ = SecondOrderTrotterOld(chain_length, J, h, lamb, final_time/layers, 1, x_basis=x_basis, barriers=barriers)
    # logical_trotter_layer_circ = FirstOrderTrotter(chain_length, J, h, lamb, final_time/layers, 1, x_basis=x_basis, barriers=barriers)
    # display(logical_trotter_layer_circ.draw(output="mpl", idle_wires=False, fold=-1))
    layout = layout[:logical_trotter_layer_circ.num_qubits] if layout is not None else None
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    physical_trotter_layer_circ = pm.run(logical_trotter_layer_circ)
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=physical_trotter_layer_circ.layout.final_index_layout())
    physical_state_preparation_circuit = pm.run(initial_state_preparation_circ)
    circs_to_return = [physical_state_preparation_circuit]
    niterations = layers // measure_every_layers
    for i in range(1, niterations + 1):
        # this_circuit = physical_state_preparation_circuit.compose(physical_trotter_layer_circ.repeat(i*measure_every_layers))
        this_circuit = physical_state_preparation_circuit.compose(physical_trotter_layer_circ.repeat(i*measure_every_layers).decompose())
        this_circuit = remove_idle_qwires(this_circuit)
        if i == 1:
            if layout is not None:
                layout_dict = {layout[i]:this_circuit.qubits[i] for i in range(this_circuit.num_qubits)}
            else:
                layout_dict = {physical_trotter_layer_circ.layout.final_index_layout()[i]:this_circuit.qubits[i] for i in range(this_circuit.num_qubits)}
            layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
            sqcancel_pm = PassManager([Optimize1qGates(target=backend.target), Optimize1qGatesDecomposition(target=backend.target), Optimize1qGatesSimpleCommutation(target=backend.target)])
            sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        this_circuit = sqopt_pm.run(this_circuit)
        circs_to_return.append(this_circuit)
    return circs_to_return

def nlayers_trotter(t, epsilon, chain_length, J, h, lamb):
    return t*(4*np.abs(J**2*lamb)/3 + 2*np.abs(J*h*lamb)/3+(h**2*lamb)/3+np.abs(J*lamb**2)/3+np.abs(h*lamb**2)/6)*(chain_length-1)/epsilon**(1/3)

def error_trotter2(t, nlayers, chain_length, J, h, lamb):
    return t**3/12/nlayers**3*(4/3*np.abs(J**2*lamb) + 2/3*np.abs(J*h*lamb) + 1/3*np.abs(J*lamb**2) + 4/6*np.abs(h*lamb**2))*(chain_length - 1)

def get_erradj_number_of_trotter_layers(times, layer_error, chain_length, J, h, lamb, max_layers=50):
    times_matrix = np.repeat(times, max_layers).reshape(len(times), max_layers)
    layers_arr = np.arange(1, max_layers + 1)
    error_trotter = error_trotter2(times_matrix, layers_arr, chain_length, J, h, lamb)
    error_hardware = 1 - (1-layer_error)**(layers_arr)
    error_total = error_trotter + error_hardware
    return np.argmin(error_total, axis=1) + 1

def erradj_particle_pair_quench_simulation_circuits_eplg(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, steps, backend, optimization_level, eplg_absolute, g=None, layout=None, x_basis=False, force_even=False, barriers=False, max_layers=100):
    layout = layout[:2*chain_length-1] if layout is not None else None
    t_arr = np.linspace(0, final_time, steps+1)
    nlayers_arr = get_erradj_number_of_trotter_layers(t_arr, eplg_absolute, chain_length, J, h, lamb, max_layers=max_layers)
    if force_even:
        odd_nlayers = ~np.equal(nlayers_arr % 2, 0)
        nlayers_arr[odd_nlayers] = nlayers_arr[odd_nlayers] + 1
    t_perlayer_arr = t_arr / nlayers_arr
    state_preparation_logical_circ = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    try:
        if backend.options.use_fractional_gates:
            sched_pm = PassManager([ALAPScheduleAnalysis(backend.instruction_durations), PadDelay(backend.instruction_durations)])
    except AttributeError:
        pass
    circs_to_return = []
    for i in range(steps):
        this_trotter_layer_logical_circuit = SecondOrderTrotter(chain_length, J, h, lamb, t_perlayer_arr[i+1], 1, g=g, x_basis=x_basis, barriers=barriers)
        this_trotter_layer_physical_circuit = trotter_pm.run(this_trotter_layer_logical_circuit)
        final_index_layout = this_trotter_layer_physical_circuit.layout.final_index_layout()
        state_prep_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=final_index_layout)
        state_preparation_physical_circuit = state_prep_pm.run(state_preparation_logical_circ)
        this_time_physical_circuit = state_preparation_physical_circuit.compose(this_trotter_layer_physical_circuit.repeat(nlayers_arr[i+1])).decompose()
        this_time_physical_circuit = remove_idle_qwires(this_time_physical_circuit, relabeling=np.arange(len(final_index_layout))[np.argsort(final_index_layout)])
        layout_dict = {final_index_layout[i]:this_time_physical_circuit.qubits[i] for i in range(this_time_physical_circuit.num_qubits)}
        layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
        sqcancel_pm = PassManager([Optimize1qGates(target=backend.target), Optimize1qGatesDecomposition(target=backend.target), Optimize1qGatesSimpleCommutation(target=backend.target)])
        try:
            if backend.options.use_fractional_gates:
                sqopt_pm = StagedPassManager(stages=["optimization", "layout", "scheduling"], layout=layout_pm, optimization=sqcancel_pm, scheduling=sched_pm)
            else:
                sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        except AttributeError:
            sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        this_time_physical_circuit = sqopt_pm.run(this_time_physical_circuit)
        if i == 0:
            measured_state_preparation_circuit = QuantumCircuit(2*chain_length-1, 2*chain_length-1)
            measured_state_preparation_circuit = measured_state_preparation_circuit.compose(state_preparation_logical_circ)
            state_preparation_physical_circuit = state_prep_pm.run(measured_state_preparation_circuit)
            state_preparation_physical_circuit.measure(np.array(final_index_layout)[np.argsort(final_index_layout)], np.argsort(np.array(final_index_layout)[np.argsort(final_index_layout)]))
            circs_to_return.append(state_preparation_physical_circuit)
        circs_to_return.append(this_time_physical_circuit)
    return circs_to_return

def erradj_particle_pair_quench_simulation_circuits_trotter(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, steps, backend, optimization_level, max_layers, error_to_adjust, g=None, layout=None, x_basis=False, force_even=False, barriers=False):
    layout = layout[:2*chain_length-1] if layout is not None else None
    t_arr = np.linspace(0, final_time, steps+1)
    last_circ_nlayers_trotter = nlayers_trotter(final_time, error_to_adjust, chain_length, J, h, lamb)
    depth_constant = max_layers/last_circ_nlayers_trotter if max_layers < last_circ_nlayers_trotter else 1
    nlayers_arr = np.ceil(depth_constant*nlayers_trotter(t_arr, error_to_adjust, chain_length, J, h, lamb)).astype(int)
    nlayers_arr[np.equal(nlayers_arr, 0)] += 1
    if force_even:
        odd_nlayers = ~np.equal(nlayers_arr % 2, 0)
        nlayers_arr[odd_nlayers] = nlayers_arr[odd_nlayers] + 1
    t_perlayer_arr = t_arr / nlayers_arr
    state_preparation_logical_circ = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    try:
        if backend.options.use_fractional_gates:
            sched_pm = PassManager([ALAPScheduleAnalysis(backend.instruction_durations), PadDelay(backend.instruction_durations)])
    except AttributeError:
        pass
    circs_to_return = []
    for i in range(steps):
        this_trotter_layer_logical_circuit = SecondOrderTrotter(chain_length, J, h, lamb, t_perlayer_arr[i+1], 1, g=g, x_basis=x_basis, barriers=barriers)
        this_trotter_layer_physical_circuit = trotter_pm.run(this_trotter_layer_logical_circuit)
        final_index_layout = this_trotter_layer_physical_circuit.layout.final_index_layout()
        state_prep_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=final_index_layout)
        state_preparation_physical_circuit = state_prep_pm.run(state_preparation_logical_circ)
        this_time_physical_circuit = state_preparation_physical_circuit.compose(this_trotter_layer_physical_circuit.repeat(nlayers_arr[i+1])).decompose()
        this_time_physical_circuit = remove_idle_qwires(this_time_physical_circuit, relabeling=np.arange(len(final_index_layout))[np.argsort(final_index_layout)])
        layout_dict = {final_index_layout[i]:this_time_physical_circuit.qubits[i] for i in range(this_time_physical_circuit.num_qubits)}
        layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
        sqcancel_pm = PassManager([Optimize1qGates(target=backend.target), Optimize1qGatesDecomposition(target=backend.target), Optimize1qGatesSimpleCommutation(target=backend.target)])
        try:
            if backend.options.use_fractional_gates:
                sqopt_pm = StagedPassManager(stages=["optimization", "layout", "scheduling"], layout=layout_pm, optimization=sqcancel_pm, scheduling=sched_pm)
            else:
                sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        except AttributeError:
            sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        this_time_physical_circuit = sqopt_pm.run(this_time_physical_circuit)
        if i == 0:
            measured_state_preparation_circuit = QuantumCircuit(2*chain_length-1, 2*chain_length-1)
            measured_state_preparation_circuit = measured_state_preparation_circuit.compose(state_preparation_logical_circ)
            state_preparation_physical_circuit = state_prep_pm.run(measured_state_preparation_circuit)
            state_preparation_physical_circuit.measure(np.array(final_index_layout)[np.argsort(final_index_layout)], np.argsort(np.array(final_index_layout)[np.argsort(final_index_layout)]))
            circs_to_return.append(state_preparation_physical_circuit)
        circs_to_return.append(this_time_physical_circuit)
    return circs_to_return

def odr_forward_circuits(chain_length, final_time, steps, backend, eplg_absolute, g=None, layout=None, x_basis=False, force_even=False, barriers=False):
    layout = layout[:2*chain_length-1] if layout is not None else None
    t_arr = np.linspace(0, final_time, steps)
    nlayers_arr = get_erradj_number_of_trotter_layers(t_arr, eplg_absolute, trotter_order=2)
    if force_even:
        odd_nlayers = ~np.equal(nlayers_arr % 2, 0)
        nlayers_arr[odd_nlayers] = nlayers_arr[odd_nlayers] + 1
    initial_pm = generate_preset_pass_manager(optimization_level=0, backend=backend, initial_layout=layout)
    first_circuit = QuantumCircuit(2*chain_length-1)
    first_circuit.measure_all()
    first_physical_circuit = initial_pm.run(first_circuit)
    circs_to_return = [first_physical_circuit]
    single_layer_cal_logical_circuit = SecondOrderTrotter(chain_length, 0, 0, 0, 0, 1, g=g, x_basis=x_basis, barriers=barriers)
    single_layer_cal_physical_circuit = initial_pm.run(single_layer_cal_logical_circuit)
    final_index_layout = single_layer_cal_physical_circuit.layout.final_index_layout()
    sqcancel_pm = PassManager([Optimize1qGates(target=backend.target), Optimize1qGatesDecomposition(target=backend.target), Optimize1qGatesSimpleCommutation(target=backend.target)])
    for i in range(steps-1):
        this_cal_circuit = single_layer_cal_physical_circuit.repeat(nlayers_arr[i+1]).decompose()
        this_cal_circuit = remove_idle_qwires(this_cal_circuit, relabeling=np.arange(len(final_index_layout))[np.argsort(final_index_layout)])
        layout_dict = {final_index_layout[i]:this_cal_circuit.qubits[i] for i in range(this_cal_circuit.num_qubits)}
        layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
        sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        this_cal_circuit = sqopt_pm.run(this_cal_circuit)
        circs_to_return.append(this_cal_circuit)
    return circs_to_return

def odr_compuncomp_circuits_eplg(chain_length, J, h, lamb, final_time, steps, backend, eplg_absolute, g=None, particle_pair_left_position=None, particle_pair_length=1, layout=None, x_basis=False, barriers=False, max_layers=50):
    layout = layout[:2*chain_length-1] if layout is not None else None
    t_arr = np.linspace(0, final_time, steps+1)
    nlayers_arr = get_erradj_number_of_trotter_layers(t_arr, eplg_absolute, chain_length, J, h, lamb, max_layers)
    odd_nlayers = ~np.equal(nlayers_arr % 2, 0)
    nlayers_arr[odd_nlayers] = nlayers_arr[odd_nlayers] + 1
    t_perlayer_arr = t_arr / nlayers_arr
    initial_pm = generate_preset_pass_manager(optimization_level=0, backend=backend, initial_layout=layout)
    if particle_pair_left_position is None:
        first_circuit = QuantumCircuit(2*chain_length-1)
        first_circuit.measure_all()
        first_physical_circuit = initial_pm.run(first_circuit)
        circs_to_return = [first_physical_circuit]
    else:
        first_circuit = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis)
        first_physical_circuit = QuantumCircuit(2*chain_length-1, 2*chain_length-1)
        first_physical_circuit.compose(first_circuit)
        first_physical_circuit = initial_pm.run(first_physical_circuit)
        final_index_layout = first_physical_circuit.layout.final_index_layout()
        first_physical_circuit.measure(np.array(final_index_layout)[np.argsort(final_index_layout)], np.argsort(np.array(final_index_layout)[np.argsort(final_index_layout)]))
        circs_to_return = [first_physical_circuit]
    sqcancel_pm = PassManager([Optimize1qGates(target=backend.target), Optimize1qGatesDecomposition(target=backend.target), Optimize1qGatesSimpleCommutation(target=backend.target)])
    for i in range(steps):
        forwards_trotter_layer = SecondOrderTrotter(chain_length, J, h, lamb, t_perlayer_arr[i+1], 1, g=g, x_basis=x_basis, barriers=barriers)
        backwards_trotter_layer = SecondOrderTrotter(chain_length, J, h, lamb, -t_perlayer_arr[i+1], 1, g=g, x_basis=x_basis, barriers=barriers)
        forwards_physical_trotter_layer = initial_pm.run(forwards_trotter_layer)
        final_index_layout = forwards_physical_trotter_layer.layout.final_index_layout()
        backwards_pm = generate_preset_pass_manager(optimization_level=0, backend=backend, initial_layout=final_index_layout)
        backwards_physical_trotter_layer = initial_pm.run(backwards_trotter_layer)
        this_time_circuit = forwards_physical_trotter_layer.repeat(nlayers_arr[i+1]//2)
        # this_time_circuit.barrier()
        this_time_circuit = this_time_circuit.compose(backwards_physical_trotter_layer.repeat(nlayers_arr[i+1]//2)).decompose()
        this_time_circuit = remove_idle_qwires(this_time_circuit, relabeling=np.arange(len(final_index_layout))[np.argsort(final_index_layout)])
        layout_dict = {final_index_layout[i]:this_time_circuit.qubits[i] for i in range(this_time_circuit.num_qubits)}
        layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
        sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        this_time_circuit = sqopt_pm.run(this_time_circuit)
        circs_to_return.append(this_time_circuit)
    return circs_to_return

def odr_compuncomp_circuits(chain_length, J, h, lamb, final_time, layers, backend, g=None, particle_pair_left_position=None, particle_pair_length=1, layout=None, x_basis=False, barriers=False):
    layout = layout[:2*chain_length-1] if layout is not None else None
    t_arr = np.linspace(0, final_time, layers+1)
    nlayers_arr = np.ceil(np.linspace(0, layers+1)).astype(int)
    nlayers_arr[np.equal(nlayers_arr % 2, 1)] += 1 
    t_perlayer_arr = t_arr / nlayers_arr
    initial_pm = generate_preset_pass_manager(optimization_level=0, backend=backend, initial_layout=layout)
    if particle_pair_left_position is None:
        first_circuit = QuantumCircuit(2*chain_length-1)
        first_circuit.measure_all()
        first_physical_circuit = initial_pm.run(first_circuit)
        circs_to_return = [first_physical_circuit]
    else:
        first_circuit = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis)
        first_physical_circuit = QuantumCircuit(2*chain_length-1, 2*chain_length-1)
        first_physical_circuit.compose(first_circuit)
        first_physical_circuit = initial_pm.run(first_physical_circuit)
        final_index_layout = first_physical_circuit.layout.final_index_layout()
        first_physical_circuit.measure(np.array(final_index_layout)[np.argsort(final_index_layout)], np.argsort(np.array(final_index_layout)[np.argsort(final_index_layout)]))
        circs_to_return = [first_physical_circuit]
    sqcancel_pm = PassManager([Optimize1qGates(target=backend.target), Optimize1qGatesDecomposition(target=backend.target), Optimize1qGatesSimpleCommutation(target=backend.target)])
    for i in range(layers):
        forwards_trotter_layer = SecondOrderTrotter(chain_length, J, h, lamb, t_perlayer_arr[i+1], 1, g=g, x_basis=x_basis, barriers=barriers)
        backwards_trotter_layer = SecondOrderTrotter(chain_length, J, h, lamb, -t_perlayer_arr[i+1], 1, g=g, x_basis=x_basis, barriers=barriers)
        forwards_physical_trotter_layer = initial_pm.run(forwards_trotter_layer)
        final_index_layout = forwards_physical_trotter_layer.layout.final_index_layout()
        backwards_pm = generate_preset_pass_manager(optimization_level=0, backend=backend, initial_layout=final_index_layout)
        backwards_physical_trotter_layer = initial_pm.run(backwards_trotter_layer)
        this_time_circuit = forwards_physical_trotter_layer.repeat(nlayers_arr[i+1]//2)
        # this_time_circuit.barrier()
        this_time_circuit = this_time_circuit.compose(backwards_physical_trotter_layer.repeat(nlayers_arr[i+1]//2)).decompose()
        this_time_circuit = remove_idle_qwires(this_time_circuit, relabeling=np.arange(len(final_index_layout))[np.argsort(final_index_layout)])
        layout_dict = {final_index_layout[i]:this_time_circuit.qubits[i] for i in range(this_time_circuit.num_qubits)}
        layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
        sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        this_time_circuit = sqopt_pm.run(this_time_circuit)
        circs_to_return.append(this_time_circuit)
    return circs_to_return