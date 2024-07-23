from qiskit.transpiler.passes import Optimize1qGates, SetLayout, ApplyLayout, DenseLayout, FullAncillaAllocation
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager, StagedPassManager, Layout
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.converters import circuit_to_instruction
from utils.circs import remove_idle_qwires
import numpy as np

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

class particle_pair_initial_state(QuantumCircuit):
    def __init__(self, chain_length, left_particle_position, particle_pair_length=1, x_basis=False):
        nqubits = 2*chain_length - 1
        super().__init__(nqubits)
        if x_basis:
            self.x(range(nqubits))
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
    total_interaction_propagator = TotalInteractionPropagator(chain_length, x_basis).decompose()
    total_interaction_propagator.assign_parameters([lamb*t_layer], inplace=True)
    total_single_body_propagator = TotalSingleBodyPropagator(chain_length, x_basis)
    total_single_body_propagator.assign_parameters([h*t_layer/2, t_layer*J/2], inplace=True)
    layer = total_single_body_propagator.compose(total_interaction_propagator).compose(total_single_body_propagator)
    if barriers: layer.barrier()
    return layer.repeat(layers).decompose()

def particle_pair_quench_simulation_circuits(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, layers, measure_every_layers=1, x_basis=False, barriers=False):
    initial_state_preparation = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    circs_to_return = [initial_state_preparation]
    ncircuits_to_iterate = layers // measure_every_layers
    for i in range(1, ncircuits_to_iterate + 1):
        this_trotter_circuit = SecondOrderTrotter(chain_length, J, h, lamb, final_time*i/ncircuits_to_iterate, i*measure_every_layers, x_basis=x_basis, barriers=barriers)
        this_complete_circuit = initial_state_preparation.compose(this_trotter_circuit)
        circs_to_return.append(this_complete_circuit)
    return circs_to_return

def physical_particle_pair_quench_simulation_circuits(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, layers, backend, optimization_level, layout=None, measure_every_layers=1, x_basis=False, barriers=False):
    initial_state_preparation_circ = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    logical_trotter_layer_circ = SecondOrderTrotter(chain_length, J, h, lamb, final_time/layers, 1, x_basis=x_basis, barriers=barriers)
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
                layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
            else:
                layout_pm = PassManager([DenseLayout(target=backend.target), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
            sqcancel_pm = PassManager([Optimize1qGates(target=backend.target)])
            sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        this_circuit = sqopt_pm.run(this_circuit)
        circs_to_return.append(this_circuit)
    return circs_to_return