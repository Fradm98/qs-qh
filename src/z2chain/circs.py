from qiskit.transpiler.passes import Optimize1qGates, SetLayout, ApplyLayout, FullAncillaAllocation, Optimize1qGatesDecomposition, Optimize1qGatesSimpleCommutation
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
            self.rx(2*t_tau, np.arange(0, self.num_qubits, 2))
            self.rx(2*t_sigma, np.arange(1, self.num_qubits, 2))
        else:
            self.rz(2*t_tau, np.arange(0, self.num_qubits, 2))
            self.rz(2*t_sigma, np.arange(1, self.num_qubits, 2))

class TotalInteractionPropagator(QuantumCircuit):
    def __init__(self, chain_length, x_basis=False):
        nqubits = 2*chain_length - 1
        qubit_list = np.arange(nqubits)
        int_qb_inds_first = np.array(qubit_list[:-1])
        int_qb_inds_second = np.array(qubit_list[1:])
        t = Parameter("t")
        super().__init__(nqubits)
        local_prop1 = LocalInteractionPropagatorFirst(x_basis)
        local_prop2 = LocalInteractionPropagatorSecond(x_basis)
        local_prop_instruction1 = circuit_to_instruction(local_prop1)
        local_prop_instruction2 = circuit_to_instruction(local_prop2)
        for qubits in int_qb_inds_first.reshape((len(int_qb_inds_first)//2, 2)):
            self.append(local_prop_instruction1, list(qubits))

        for qubits in int_qb_inds_second.reshape((len(int_qb_inds_second)//2, 2)):
            self.append(local_prop_instruction2, list(qubits))

        if x_basis:
            self.rz(2*t, np.arange(1, nqubits, 2))
        else:
            self.rx(2*t, np.arange(1, nqubits, 2))

        for qubits in int_qb_inds_second.reshape((len(int_qb_inds_second)//2, 2)):
            self.append(local_prop_instruction2, list(qubits))

        for qubits in int_qb_inds_first.reshape((len(int_qb_inds_first)//2, 2)):
            self.append(local_prop_instruction1, list(qubits))

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

def SecondOrderTrotter(chain_length, J, h, lamb, t_total, layers, x_basis=False, barriers=False):
    t_layer = t_total/layers
    # total_interaction_propagator = TotalInteractionPropagator(chain_length, x_basis)
    total_interaction_propagator = TotalInteractionPropagator(chain_length, x_basis).decompose()
    total_interaction_propagator.assign_parameters([lamb*t_layer], inplace=True)
    total_single_body_propagator = TotalSingleBodyPropagator(chain_length, x_basis)
    total_single_body_propagator.assign_parameters([h*t_layer/2, t_layer*J/2], inplace=True)
    layer = total_single_body_propagator.compose(total_interaction_propagator).compose(total_single_body_propagator)
    if barriers: layer.barrier()
    # return layer.repeat(layers)
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
    # return layer.repeat(layers)
    return layer.repeat(layers).decompose()

def particle_pair_quench_simulation_circuits(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, layers, measure_every_layers=1, x_basis=False, barriers=False):
    initial_state_preparation = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    circs_to_return = [initial_state_preparation]
    ncircuits_to_iterate = layers // measure_every_layers
    sqcancel_pm = PassManager([Optimize1qGatesDecomposition()])
    sqopt_pm = StagedPassManager(stages=["optimization"], optimization=sqcancel_pm)
    for i in range(1, ncircuits_to_iterate + 1):
        this_trotter_circuit = SecondOrderTrotter(chain_length, J, h, lamb, final_time*i/ncircuits_to_iterate, i*measure_every_layers, x_basis=x_basis, barriers=barriers)
        this_complete_circuit = initial_state_preparation.compose(this_trotter_circuit)
        this_complete_circuit = sqopt_pm.run(this_complete_circuit)
        circs_to_return.append(this_complete_circuit)
    return circs_to_return

def physical_particle_pair_quench_simulation_circuits(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, layers, backend, optimization_level, layout=None, measure_every_layers=1, x_basis=False, barriers=False):
    initial_state_preparation_circ = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    logical_trotter_layer_circ = SecondOrderTrotter(chain_length, J, h, lamb, final_time/layers, 1, x_basis=x_basis, barriers=barriers)
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

def get_erradj_number_of_trotter_layers(times, eplg_absolute, trotter_order, max_layers=100):
    times_matrix = np.repeat(times, max_layers).reshape(len(times), max_layers)
    layers_arr = np.arange(1, max_layers + 1)
    error_trotter = (times_matrix/layers_arr)**(trotter_order + 1)
    error_hardware = 1 - (1-eplg_absolute)**layers_arr
    error_total = error_trotter + error_hardware
    return np.argmin(error_total, axis=1) + 1

def erradj_particle_pair_quench_simulation_circuits(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, steps, backend, optimization_level, eplg_absolute, layout=None, x_basis=False, barriers=False):
    layout = layout[:2*chain_length-1] if layout is not None else None
    t_arr = np.linspace(0, final_time, steps)
    nlayers_arr = get_erradj_number_of_trotter_layers(t_arr, eplg_absolute, trotter_order=2)
    t_perlayer_arr = t_arr / nlayers_arr
    # First compilation to set layout
    trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    first_trotter_layer_logical_circuit = SecondOrderTrotter(chain_length, J, h, lamb, t_perlayer_arr[1], 1, x_basis=x_basis, barriers=barriers)
    first_trotter_layer_physical_circuit = trotter_pm.run(first_trotter_layer_logical_circuit)
    state_preparation_logical_circ = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    state_prep_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=first_trotter_layer_physical_circuit.layout.final_index_layout())
    state_preparation_physical_circuit = state_prep_pm.run(state_preparation_logical_circ)
    first_time_physical_circuit = state_preparation_physical_circuit.compose(first_trotter_layer_physical_circuit.repeat(nlayers_arr[1])).decompose()
    first_time_physical_circuit = remove_idle_qwires(first_time_physical_circuit)
    if layout is not None:
        layout_dict = {layout[i]:first_time_physical_circuit.qubits[i] for i in range(first_time_physical_circuit.num_qubits)}
    else:
        layout_dict = {first_trotter_layer_physical_circuit.layout.final_index_layout()[i]:first_time_physical_circuit.qubits[i] for i in range(first_time_physical_circuit.num_qubits)}
    layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
    sqcancel_pm = PassManager([Optimize1qGates(target=backend.target), Optimize1qGatesDecomposition(target=backend.target), Optimize1qGatesSimpleCommutation(target=backend.target)])
    sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
    first_time_physical_circuit = sqopt_pm.run(first_time_physical_circuit)
    circs_to_return = [state_preparation_physical_circuit, first_time_physical_circuit]
    for i in range(2, steps):
        this_logical_trotter_layer = SecondOrderTrotter(chain_length, J, h, lamb, t_perlayer_arr[i], 1, x_basis=x_basis, barriers=barriers)
        this_physical_trotter_layer = trotter_pm.run(this_logical_trotter_layer)
        this_physical_circuit = state_preparation_physical_circuit.compose(this_physical_trotter_layer.repeat(nlayers_arr[i])).decompose()
        this_physical_circuit = remove_idle_qwires(this_physical_circuit)
        this_physical_circuit = sqopt_pm.run(this_physical_circuit)
        circs_to_return.append(this_physical_circuit)
    return circs_to_return