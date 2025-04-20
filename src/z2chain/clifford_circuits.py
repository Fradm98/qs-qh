from qiskit.converters import circuit_to_instruction, dag_to_circuit, circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit import QuantumCircuit, qpy
from qiskit.circuit.library import RZGate
from qiskit.quantum_info import Clifford
from random import randrange
import numpy as np

rand_angles = (0, np.pi / 2, np.pi, 3 * np.pi / 2)

def load_circs(filepath):
    with open(filepath, "rb") as f:
        circs = qpy.load(f)
    return circs

def _random_z():
    return RZGate(rand_angles[randrange(4)])

def _is_clifford(gate):
    try:
        Clifford(gate)
        return True
    except:
        return False

def generate_random_non_clifford_subset(non_clifford_register: list, N: int):
    new_idxs = []
    new_non_clifford_register = []
    idxs = np.random.choice(range(0, len(non_clifford_register)), N, replace=False)
    for idx in idxs:
        new_non_clifford_register.append(non_clifford_register[idx])
        new_idxs.append(non_clifford_register[idx][0])
    return new_idxs, new_non_clifford_register

def near_clifford_circuit(circ: QuantumCircuit, N: int):
    """
    near clifford circuit

    This function takes a logical quantum circuit (with many clifford gates), and
    makes it near clifford, limiting the number of clifford gate to N.
    This is done by cliffordize the whole circuit and subsequently peaking a 
    random subset N of non-clifford gates in the original circuit putting them 
    back in place.
    
    circ: QuantumCircuit - quantum non-clifford circuit
    N: int - subset of non-clifford operations in the cliffordized circuit

    """
    dag = circuit_to_dag(circ)

    non_clifford_register = []
    for i, node in enumerate(dag.op_nodes()):

        if _is_clifford(node.op) == False:
            try:
                old_node_op = node.op
                dag.substitute_node(node, _random_z(), inplace=True)
                non_clifford_register.append((i, old_node_op))
            except:
                print('Keep: ', node.op)

    new_idxs, new_non_clifford_register = generate_random_non_clifford_subset(non_clifford_register, N=N)
    ratio = N/len(dag.op_nodes())
    j = 0
    for i, node in enumerate(dag.op_nodes()):

        if i in new_idxs:
            dag.substitute_node(node, new_non_clifford_register[j][1], inplace=True)
            # print('sub: ', new_non_clifford_register[j])
            j += 1

    near_clifford_circ = dag_to_circuit(dag)
    return near_clifford_circ, ratio

def near_clifford_circuits_set(circ: QuantumCircuit, n: int, N: int):
    """
    near clifford circuits set

    This function generates n near clifford circuits to form a set.
    Each near clifford circuit has N non-clifford gates peaked randomly.

    circ: QuantumCircuit - quantum non-clifford circuit
    n: int - set of near clifford circuits
    N: int - subset of non-clifford operations in the cliffordized circuit

    """
    set = []
    for rep in range(n):
        near_clifford, ratio = near_clifford_circuit(circ, N)
        set.append(near_clifford)

    print(f"Near-Clifford set has {ratio*100:.2f}% of Non-clifford gates")
    
    return set
