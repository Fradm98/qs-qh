from qiskit.transpiler.passes import Optimize1qGates, SetLayout, ApplyLayout, DenseLayout, FullAncillaAllocation
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager, StagedPassManager, Layout
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.converters import circuit_to_instruction
from utils.circs import remove_idle_qwires
from qiskit import QuantumRegister
import numpy as np

class LocalInteractionPropagator(QuantumCircuit):
    def __init__(self):
        super().__init__(3)
        t = Parameter("t")
        self.cx(1, 0)
        self.cx(1, 2)
        self.rx(2*t, 1)
        self.cx(1, 2)
        self.cx(1, 0)

class LowDepthLocalInteractionPropagatorFirst(QuantumCircuit):
    def __init__(self):
        super().__init__(2)
        self.cx(1, 0)
        
class LowDepthLocalInteractionPropagatorSecond(QuantumCircuit):
    def __init__(self):
        super().__init__(2)
        self.cx(0, 1)

class TotalInteractionPropagator(QuantumCircuit):
    def __init__(self, chain_length):
        nqubits = 2*chain_length - 1
        qubit_list = np.arange(nqubits)
        int_qb_inds_first = qubit_list[np.less(qubit_list % 4, 3)][:-1 if chain_length % 2 else None]
        int_qb_inds_second = qubit_list[2::][np.less((qubit_list[2::] - 2) % 4, 3)][:None if chain_length % 2 else -1]
        super().__init__(nqubits)
        local_prop = LocalInteractionPropagator()
        local_prop_instruction = circuit_to_instruction(local_prop)
        for qubits in int_qb_inds_first.reshape((len(int_qb_inds_first)//3, 3)):
            self.append(local_prop_instruction, list(qubits))
        for qubits in int_qb_inds_second.reshape((len(int_qb_inds_second)//3, 3)):
            self.append(local_prop_instruction, list(qubits))

class LowDepthTotalInteractionPropagator(QuantumCircuit):
    def __init__(self, chain_length):
        nqubits = 2*chain_length - 1
        qubit_list = np.arange(nqubits)
        int_qb_inds_first = np.array(qubit_list[:-1])
        int_qb_inds_second = np.array(qubit_list[1:])
        t = Parameter("t")
        super().__init__(nqubits)
        local_prop1 = LowDepthLocalInteractionPropagatorFirst()
        local_prop2 = LowDepthLocalInteractionPropagatorSecond()
        local_prop_instruction1 = circuit_to_instruction(local_prop1)
        local_prop_instruction2 = circuit_to_instruction(local_prop2)
        for qubits in int_qb_inds_first.reshape((len(int_qb_inds_first)//2, 2)):
            self.append(local_prop_instruction1, list(qubits))

        for qubits in int_qb_inds_second.reshape((len(int_qb_inds_second)//2, 2)):
            self.append(local_prop_instruction2, list(qubits))

        for qubits in range(nqubits):
            if qubits%2 == 1:
                self.rx(2*t, qubits)

        for qubits in int_qb_inds_second.reshape((len(int_qb_inds_second)//2, 2)):
            self.append(local_prop_instruction2, list(qubits))

        for qubits in int_qb_inds_first.reshape((len(int_qb_inds_first)//2, 2)):
            self.append(local_prop_instruction1, list(qubits))

class TotalSingleBodyPropagator(QuantumCircuit):
    def __init__(self, chain_length):
        nqubits = 2*chain_length - 1
        super().__init__(nqubits)
        t_tau = Parameter("t_τ")
        t_sigma = Parameter("t_σ")
        self.rz(2*t_tau, np.arange(0, self.num_qubits, 2))
        self.rz(2*t_sigma, np.arange(1, self.num_qubits, 2))

def FirstOrderTrotter(chain_length, J, h, lamb, t_total, layers, sqrot_first=False, barriers=False):
    t_layer = t_total/layers
    total_interaction_propagator = TotalInteractionPropagator(chain_length).decompose()
    total_interaction_propagator.assign_parameters([lamb*t_layer], inplace=True)
    total_single_body_propagator = TotalSingleBodyPropagator(chain_length)
    total_single_body_propagator.assign_parameters([h*t_layer, t_layer*J], inplace=True)
    if sqrot_first:
        layer = total_single_body_propagator.compose(total_interaction_propagator)
    else:
        layer = total_interaction_propagator.compose(total_single_body_propagator)
    if barriers:
        layer.barrier()
    return layer.repeat(layers).decompose()

def SecondOrderTrotter(chain_length, J, h, lamb, t_total, layers, barriers=False):
    t_layer = t_total/layers
    total_interaction_propagator = TotalInteractionPropagator(chain_length).decompose()
    total_interaction_propagator.assign_parameters([lamb*t_layer], inplace=True)
    total_single_body_propagator = TotalSingleBodyPropagator(chain_length)
    total_single_body_propagator.assign_parameters([h*t_layer/2, t_layer*J/2], inplace=True)
    layer = total_single_body_propagator.compose(total_interaction_propagator).compose(total_single_body_propagator)
    if barriers: layer.barrier()
    return layer.repeat(layers).decompose()

def LowDepthSecondOrderTrotter(chain_length, J, h, lamb, t_total, layers, barriers=False):
    t_layer = t_total/layers
    total_interaction_propagator = LowDepthTotalInteractionPropagator(chain_length).decompose()
    total_interaction_propagator.assign_parameters([lamb*t_layer], inplace=True)
    total_single_body_propagator = TotalSingleBodyPropagator(chain_length)
    total_single_body_propagator.assign_parameters([h*t_layer/2, t_layer*J/2], inplace=True)
    layer = total_single_body_propagator.compose(total_interaction_propagator).compose(total_single_body_propagator)
    if barriers: layer.barrier()
    return layer.repeat(layers).decompose()

class particle_pair_initial_state(QuantumCircuit):
    def __init__(self, chain_length, left_particle_position, particle_pair_length=1):
        nqubits = 2*chain_length - 1
        super().__init__(nqubits)
        self.x(range(2*left_particle_position, 2*(left_particle_position + particle_pair_length) + 1))

def particle_pair_quench_simulation_circuits(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, layers, measure_every_layers=1, barriers=False):
    initial_state_preparation = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length)
    circs_to_return = [initial_state_preparation]
    ncircuits_to_iterate = layers // measure_every_layers
    for i in range(1, ncircuits_to_iterate + 1):
        this_trotter_circuit = SecondOrderTrotter(chain_length, J, h, lamb, final_time*i/ncircuits_to_iterate, i*measure_every_layers, barriers=barriers)
        this_complete_circuit = initial_state_preparation.compose(this_trotter_circuit)
        circs_to_return.append(this_complete_circuit)
    return circs_to_return

def physical_particle_pair_quench_simulation_circuits(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, layers, backend, optimization_level, layout=None, measure_every_layers=1, depth="low", barriers=False):
    initial_state_preparation_circ = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length)
    if depth == "low":
        logical_trotter_layer_circ = LowDepthSecondOrderTrotter(chain_length, J, h, lamb, final_time/layers, 1, barriers)
    else:
        logical_trotter_layer_circ = SecondOrderTrotter(chain_length, J, h, lamb, final_time/layers, 1, barriers)
    layout = layout[:logical_trotter_layer_circ.num_qubits] if layout is not None else None
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    physical_state_preparation_circuit = pm.run(initial_state_preparation_circ)
    physical_trotter_layer_circ = pm.run(logical_trotter_layer_circ)
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