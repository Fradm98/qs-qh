from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
import numpy as np

class ExampleCircuit(QuantumCircuit):
    """Parameterized MBL non-periodic chain (i.e. 1D) quantum circuit.

    Parameters correspond to interaction strength (θ), and
    disorders vector (φ) with one entry per qubit. In 1D,
    θ < 0.16π ≈ 0.5 corresponds to the MBL regime; beyond such
    critical value the dynamics become ergodic (i.e. thermal).
    Disorders are random on a qubit by qubit basis [1].
    
    Args:
        num_qubits: number of qubits (must be even).
        depth: two-qubit depth.
        barriers: if True adds barriers between layers

    Notes:
        [1] Shtanko et.al. Uncovering Local Integrability in Quantum
            Many-Body Dynamics, https://arxiv.org/abs/2307.07552
    """

    def __init__(self, num_qubits: int, depth: int, *, barriers: bool = False) -> None:
        super().__init__(num_qubits)

        theta = Parameter("θ")
        phis = ParameterVector("φ", num_qubits)
        
        self.x(range(1, num_qubits, 2))
        for layer in range(depth):
            layer_parity = layer % 2
            if barriers and layer > 0:
                self.barrier()
            for qubit in range(layer_parity, num_qubits - 1, 2):
                self.cz(qubit, qubit + 1)
                self.u(theta, 0, np.pi, qubit)
                self.u(theta, 0, np.pi, qubit + 1)
            for qubit in range(num_qubits):
                self.p(phis[qubit], qubit)

def cunc_example_circuit(nqubits, twoqdepth, seed, disorder_parameter):
    if twoqdepth % 4 != 0:
        raise ValueError("Two-qubit depth must be a multiple of 4")
    
    forward_circ = ExampleCircuit(nqubits, twoqdepth // 2, barriers=False)
    inv_cirv = forward_circ.inverse()
    forward_circ.barrier()
    circ = forward_circ.compose(inv_cirv)

    rng = np.random.default_rng(seed=seed)
    parameter_values = rng.uniform(-np.pi, np.pi, size=circ.num_parameters)
    parameter_values[0] = disorder_parameter
    circ.assign_parameters(parameter_values, inplace=True)
    
    return circ

def single_pauliz_average_op(nqubits):
    paulistrs = ["I"*i + "Z" + "I"*(nqubits - 1 - i) for i in range(nqubits)]
    coeffs = 1 / nqubits
    logical_observable = SparsePauliOp(paulistrs, coeffs)
    return logical_observable

def adjacent_pauliz_pairs_average_op(nqubits):
    paulistrs = ["I"*i + "ZZ" + "I"*(nqubits - 2 - i) for i in range(nqubits - 1)]
    coeffs = 1 / (nqubits - 1)
    logical_observable = SparsePauliOp(paulistrs, coeffs)
    return logical_observable