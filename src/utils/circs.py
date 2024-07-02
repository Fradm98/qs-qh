from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import QuantumRegister

def remove_idle_qwires(circ):
    dag = circuit_to_dag(circ)

    idle_wires = list(dag.idle_wires())
    for w in idle_wires:
        dag._remove_idle_wire(w)
        dag.qubits.remove(w)

    for qreg in dag.qregs.values():
        dag.remove_qregs(qreg)

    qreg = QuantumRegister(name="q", bits=dag.qubits)
    dag.add_qreg(qreg)

    return dag_to_circuit(dag)