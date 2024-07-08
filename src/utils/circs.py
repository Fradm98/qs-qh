from qiskit.converters import circuit_to_instruction, dag_to_circuit, circuit_to_dag
from qiskit import QuantumCircuit
import numpy as np

def remove_idle_qwires(circ):
    dag = circuit_to_dag(circ)

    idle_wires = list(dag.idle_wires())
    num_non_idle = dag.num_qubits() - len(idle_wires)
    for w in idle_wires:
        dag._remove_idle_wire(w)
        dag.qubits.remove(w)

    for qreg in dag.qregs.values():
        dag.remove_qregs(qreg)

    layout = circ.layout.final_index_layout() if circ.layout is not None else list(range(num_non_idle))
    qubits_to_apply = list(np.argsort(layout))
    circ_without_idle = dag_to_circuit(dag)
    circ_instruction = circuit_to_instruction(circ_without_idle)
    to_return = QuantumCircuit(num_non_idle)
    to_return.compose(circ_instruction, qubits=qubits_to_apply, inplace=True)
    return to_return.decompose()

def check_and_measure_active_qubits(circuit):
    if not circuit.cregs:
        circuit.measure_active()
    return circuit