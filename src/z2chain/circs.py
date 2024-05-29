from qiskit.circuit import QuantumCircuit, Parameter, InstructionSet
from qiskit.converters import circuit_to_instruction
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
    total_interaction_propagator.assign_parameters([2*lamb/J*t_layer], inplace=True)
    total_single_body_propagator = TotalSingleBodyPropagator(chain_length)
    total_single_body_propagator.assign_parameters([2*t_layer, 2*h/J*t_layer], inplace=True)
    if sqrot_first:
        layer = total_single_body_propagator.compose(total_interaction_propagator)
    else:
        layer = total_interaction_propagator.compose(total_single_body_propagator)
    if barriers:
        layer.barrier()
    return layer.repeat(layers).decompose()