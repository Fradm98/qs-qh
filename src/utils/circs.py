from qiskit.transpiler.passes import Optimize1qGates, SetLayout, ApplyLayout, FullAncillaAllocation, Optimize1qGatesDecomposition, Optimize1qGatesSimpleCommutation
from qiskit.converters import circuit_to_instruction, dag_to_circuit, circuit_to_dag
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager, StagedPassManager, Layout
from qiskit.dagcircuit import DAGCircuit
from qiskit import QuantumCircuit
import qiskit.qasm3 as qasm3
import numpy as np

def count_non_idle_qubits(circ):
    if type(circ) is not DAGCircuit:
        dag = circuit_to_dag(circ)
    idle_wires = list(dag.idle_wires())
    return dag.num_qubits() - len(idle_wires)

def remove_idle_qwires(circ, relabeling=None):
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
    qc_args = (num_non_idle, num_non_idle) if circ.num_clbits > 0 else (num_non_idle,)
    to_return = QuantumCircuit(*qc_args)
    to_return = to_return.compose(circ_instruction, qubits=qubits_to_apply).decompose()
    if relabeling is not None:
        before_relabeling = to_return.copy()
        qc_args = (before_relabeling.num_qubits, before_relabeling.num_qubits) if before_relabeling.num_clbits > 0 else (before_relabeling.num_qubits,)
        to_return = QuantumCircuit(*qc_args)
        measurement_operations = []
        for instruction_data in before_relabeling.data:
            if instruction_data.operation.name == "measure":
                measurement_operations.append(instruction_data)
            else:
                to_return.append(instruction_data.operation, qargs=[relabeling[qubit._index] for qubit in instruction_data.qubits])
        # Place the measurement operations at the end
        for measurement_instruction in measurement_operations:
            new_indices = [relabeling[qubit._index] for qubit in instruction_data.qubits]
            to_return.append(measurement_instruction.operation, qargs=new_indices, cargs=new_indices)
        
    return to_return

def check_and_measure_active_qubits(circuit):
    if not circuit.cregs:
        circuit = circuit.copy()
        circuit.measure_active()
    return circuit

def compute_uncompute(original_circuit, barrier=False):
    to_return = original_circuit.copy()
    if barrier: to_return.barrier()
    return to_return.compose(original_circuit.inverse())

def simplify_logical_circuits(circuits, optimization_level=3):
    pm = generate_preset_pass_manager(optimization_level=optimization_level)
    return pm.run(circuits)

def join_transpiled_circuits(*circuits, layout_circ=0, backend=None):
    layout = circuits[layout_circ].layout.final_index_layout()
    joined_circuit = QuantumCircuit(circuits[layout_circ].num_qubits, circuits[layout_circ].num_clbits)
    for circuit in circuits:
        joined_circuit = joined_circuit.compose(circuit)
    # Reorder measurements
    reordered_circ = QuantumCircuit(joined_circuit.num_qubits, joined_circuit.num_clbits)
    measurements = []
    for instruction_data in joined_circuit.data:
        if instruction_data.operation.name == "measure":
            measurements.append(instruction_data)
        else:
            reordered_circ.append(instruction_data.operation, qargs=[qubit._index for qubit in instruction_data.qubits])
    for measurement_instruction in measurements:
        indices = [qubit._index for qubit in measurement_instruction.qubits]
        reordered_circ.append(measurement_instruction.operation, qargs=indices, cargs=indices)
    if backend is not None:
        joined_circuit = remove_idle_qwires(reordered_circ, relabeling=np.arange(len(layout))[np.argsort(layout)])
        layout_dict = {layout[i]:reordered_circ.qubits[i] for i in range(joined_circuit.num_qubits)}
        layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
        sqcancel_pm = PassManager([Optimize1qGates(target=backend.target), Optimize1qGatesDecomposition(target=backend.target), Optimize1qGatesSimpleCommutation(target=backend.target)])
        sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        reordered_circ = sqopt_pm.run(reordered_circ)
        return reordered_circ
    else:
        return reordered_circ
    
def append_basis_change_circuit(circuit, basis, backend=None):
    nqubits = count_non_idle_qubits(circuit)
    change_of_basis = QuantumCircuit(nqubits)
    if basis == "Y":
        change_of_basis.sdg(np.arange(nqubits))
        change_of_basis.h(np.arange(nqubits))
    if basis == "X":
        change_of_basis.h(np.arange(nqubits))
    if (backend is not None) and (circuit.layout is not None):
        pm = generate_preset_pass_manager(backend=backend, initial_layout=circuit.layout.final_index_layout(), optimization_level=0)
        change_of_basis = pm.run(change_of_basis)
    return join_transpiled_circuits(circuit, change_of_basis, backend=backend)

def depth2qb(circ):
    return circ.depth(lambda x: len(x.qubits)>1)