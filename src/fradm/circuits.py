from numpy import pi
from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit

class MBLCircuit(QuantumCircuit):
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
                self.u(theta, 0, pi, qubit)
                self.u(theta, 0, pi, qubit + 1)
            for qubit in range(num_qubits):
                self.p(phis[qubit], qubit)

class Z2MassiveChainCircuit(QuantumCircuit):
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

    def __init__(self, num_qubits: int, depth: int, delta: float, h1: float=None, h2: float=None, barriers: bool = False) -> None:
        super().__init__(num_qubits)
        self.h1 = h1
        self.h2 = h2

        theta = Parameter("θ")
        # phis = ParameterVector("φ", num_qubits)
        
        for layer in range(depth):
            # matter
            self.rz(phi=theta, qubit=range(0, num_qubits, 2))
            # gauge
            self.rz(phi=theta*self.h1, qubit=range(1, num_qubits, 2))
            if barriers:
                self.barrier()
            # matter-gauge interaction
            self.h(qubit=range(num_qubits))
            self.cx(control_qubit=range(0,num_qubits-1), target_qubit=range(1,num_qubits))
            self.rz(phi=theta*self.h2, qubit=num_qubits-1)
            self.cx(control_qubit=range(num_qubits-2,-1,-1), target_qubit=range(num_qubits-1,0,-1))
            self.h(qubit=range(num_qubits))
            if barriers:
                self.barrier()