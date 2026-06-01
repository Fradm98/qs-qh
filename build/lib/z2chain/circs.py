from qiskit_ibm_runtime.transpiler.passes.scheduling import ALAPScheduleAnalysis, PadDelay
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from utils.paulis import PauliTensor, PauliTensorSum, commutator
from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.circuit import QuantumCircuit, Parameter
from sympy.physics.paulialgebra import Pauli
from multiprocessing import Pool, cpu_count
from qiskit_aer import AerSimulator
from sympy.core.numbers import One
import utils.multiproductf as mpf
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
    def __init__(self, chain_length, Js, hs, t, x_basis=False):
    # def __init__(self, chain_length, x_basis=False):
        nqubits = 2*chain_length - 1
        super().__init__(nqubits)
        Js = np.array(Js)
        hs = np.array(hs)
        if len(Js.shape) > 1:
            raise ValueError(f"Js should be a scalar or an array of length chain_length")
        elif len(Js.shape) == 1:
            if len(Js) == 1:
                raise ValueError(f"Js should be a scalar or an array of length chain_length")
        else:
            Js = np.full(chain_length, Js)
        if len(hs.shape) > 1:
            raise ValueError(f"hs should be a scalar or an array of length chain_length - 1")
        elif len(hs.shape) == 1:
            if len(hs) != (chain_length - 1):
                raise ValueError(f"hs should be a scalar or an array of length chain_length - 1")
        else:
            hs = np.full(chain_length - 1, hs)

        # t_tau = Parameter("t_tau")
        # t_sigma = Parameter("t_sigma")
        # if x_basis:
        #     self.rx(-2*t_tau, np.arange(0, self.num_qubits, 2))
        #     self.rx(-2*t_sigma, np.arange(1, self.num_qubits, 2))
        # else:
        #     self.rz(-2*t_tau, np.arange(0, self.num_qubits, 2))
        #     self.rz(-2*t_sigma, np.arange(1, self.num_qubits, 2))

        for i, qbind in enumerate(range(0, nqubits, 2)):
            if x_basis:
                self.rx(-2*Js[i]*t, qbind)
            else:
                self.rz(-2*Js[i]*t, qbind)
        for i, qbind in enumerate(range(1, nqubits, 2)):
            if x_basis:
                self.rx(-2*hs[i]*t, qbind)
            else:
                self.rz(-2*hs[i]*t, qbind)

class TotalInteractionPropagator(QuantumCircuit):
    def __init__(self, chain_length, lambdas, t, x_basis=False):
    # def __init__(self, chain_length, x_basis=False):
        nqubits = 2*chain_length - 1
        lambdas = np.array(lambdas)
        if len(lambdas.shape) > 1:
            raise ValueError("The lambdas parameter must be a vector or scalar")
        elif len(lambdas.shape) == 1:
            if len(lambdas) != len(chain_length-1):
                raise ValueError("The length of the lambdas vector must be equal to the number of edges in the lattice")
        else:
            lambdas = np.repeat(lambdas, chain_length-1)
        first_cnots_trgt_qubits = np.arange(0, nqubits-1, 2)
        first_cnots_ctrl_qubits = np.arange(1, nqubits, 2)
        second_cnots_trgt_qubits = np.arange(2, nqubits, 2)
        super().__init__(nqubits)
        if x_basis:
            self.cx(first_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.cx(second_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            for i, qind in enumerate(range(1, nqubits, 2)):
                self.rz(-2*lambdas[i]*t, qind)
            self.cx(second_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.cx(first_cnots_trgt_qubits, first_cnots_ctrl_qubits)
        else:
            self.cx(first_cnots_ctrl_qubits, first_cnots_trgt_qubits)
            self.cx(first_cnots_ctrl_qubits, second_cnots_trgt_qubits)
            for i, qind in enumerate(range(1, nqubits, 2)):
                self.rx(-2*lambdas[i]*t, qind)
            self.cx(first_cnots_ctrl_qubits, second_cnots_trgt_qubits)
            self.cx(first_cnots_ctrl_qubits, first_cnots_trgt_qubits)

class TotalInteractionPropagatorDD(QuantumCircuit):
    def __init__(self, chain_length, lambdas, t_g, t, x_basis=False):
        nqubits = 2*chain_length - 1
        first_cnots_trgt_qubits = np.arange(0, nqubits-1, 2)
        first_cnots_ctrl_qubits = np.arange(1, nqubits, 2)
        second_cnots_trgt_qubits = np.arange(2, nqubits, 2)
        lambdas = np.array(lambdas)
        if len(lambdas.shape) > 1:
            raise ValueError("The lambdas parameter must be a vector or scalar")
        elif len(lambdas.shape) == 1:
            if len(lambdas) != (chain_length - 1):
                raise ValueError("The length of the lambdas vector must be equal to the number of edges in the lattice")
        else:
            lambdas = np.full(chain_length - 1, lambdas)
        t_g = np.array(t_g)
        if t_g.ndim == 0:
            t_g = np.full(chain_length, t_g)
        elif t_g.ndim > 1:
            raise ValueError(f"t_g should be a scalar or an array of shape ({chain_length},)")
        else:
            if t_g.shape[0] != chain_length:
                raise ValueError(f"t_g should be a scalar or an array of shape ({chain_length},)")
        super().__init__(nqubits)
        if x_basis:
            self.cx(first_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.cx(second_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            for i, qind in enumerate(range(0, nqubits, 2)):
                self.rx(-2*t_g[i], qind)
            for i, qind in enumerate(range(1, nqubits, 2)):
                self.rz(-2*lambdas[i]*t, qind)
            self.cx(second_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.cx(first_cnots_trgt_qubits, first_cnots_ctrl_qubits)
        else:
            self.cx(first_cnots_ctrl_qubits, first_cnots_trgt_qubits)
            self.cx(first_cnots_ctrl_qubits, second_cnots_trgt_qubits)
            for i, qind in enumerate(range(0, nqubits, 2)):
                self.rz(-2*t_g[i], qind)
            for i, qbind in enumerate(range(1, nqubits, 2)):
                self.rx(-2*lambdas[i]*t, qbind)
            self.cx(first_cnots_ctrl_qubits, second_cnots_trgt_qubits)
            self.cx(first_cnots_ctrl_qubits, first_cnots_trgt_qubits)

class particle_pair_initial_state(QuantumCircuit):
    def __init__(self, chain_length, left_particle_position, particle_pair_length=1, x_basis=False):
        nqubits = 2*chain_length - 1
        super().__init__(nqubits)
        gate_qubits = [2*left_particle_position] + list(range(2*left_particle_position+1, 2*(left_particle_position + particle_pair_length), 2)) + [2*(left_particle_position + particle_pair_length)]
        if particle_pair_length > 0:
            if x_basis:
                self.z(gate_qubits)
            else:
                self.x(gate_qubits)

class single_particle_initial_state(QuantumCircuit):
    def __init__(self, chain_length, particle_position, x_basis=False):
        nqubits = 2*chain_length - 1
        super().__init__(nqubits)
        gate_qubits = [2*particle_position] + list(range(2*particle_position+1, nqubits, 2))
        if x_basis:
            self.z(gate_qubits)
        else:
            self.x(gate_qubits)

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

def SecondOrderTrotter(chain_length, J, h, lamb, t_total, layers, g=None, x_basis=False, barriers=False):
    t_layer = t_total/layers
    if g is None:
        total_interaction_propagator = TotalInteractionPropagator(chain_length, lamb, t_layer, x_basis).decompose()
    else:
        total_interaction_propagator = TotalInteractionPropagatorDD(chain_length, lamb, g, t_layer, x_basis=x_basis)
    total_single_body_propagator = TotalSingleBodyPropagator(chain_length, J, h, t_layer/2, x_basis)
    layer = total_single_body_propagator.compose(total_interaction_propagator).compose(total_single_body_propagator)
    if barriers: layer.barrier()
    return layer.repeat(layers).decompose()

def particle_pair_quench_simulation_circuits(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, layers, g=None, measure_every_layers=1, x_basis=False, barriers=False):
    initial_state_preparation = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    circs_to_return = [initial_state_preparation]
    ncircuits_to_iterate = layers // measure_every_layers
    sqcancel_pm = PassManager([Optimize1qGatesDecomposition()])
    sqopt_pm = StagedPassManager(stages=["optimization"], optimization=sqcancel_pm)
    for i in range(1, ncircuits_to_iterate + 1):
        this_trotter_circuit = SecondOrderTrotter(chain_length, J, h, lamb, final_time*i/ncircuits_to_iterate, i*measure_every_layers, g=g, x_basis=x_basis, barriers=barriers)
        this_complete_circuit = initial_state_preparation.compose(this_trotter_circuit)
        this_complete_circuit = sqopt_pm.run(this_complete_circuit)
        circs_to_return.append(this_complete_circuit)
    return circs_to_return

# --------------------------- 
#       TROTTER ERROR
# ---------------------------

def hamiltonian_interaction_terms(chain_length, lamb):
    each_interaction_qubits = np.zeros((chain_length-1, 3), dtype=int)
    each_interaction_qubits[:, 0] = np.arange(0, 2*chain_length-2, 2)
    each_interaction_qubits[:, 1] = np.arange(0, 2*chain_length-2, 2) + 1
    each_interaction_qubits[:, 2] = np.arange(0, 2*chain_length-2, 2) + 2
    all_qubits = np.arange(2*chain_length-1)
    tensors_to_sum = [PauliTensor([Pauli(1) if j in this_int_qubit else One for j in all_qubits], coefficient=-lamb) for this_int_qubit in each_interaction_qubits]
    return PauliTensorSum(*tensors_to_sum)

def hamiltonian_single_terms(chain_length, J, h):
    all_qubits = np.arange(2*chain_length-1)
    tensors_to_sum = [PauliTensor([Pauli(3) if j == this_qubit else One for j in all_qubits], coefficient=-J if this_qubit % 2 == 0 else -h) for this_qubit in all_qubits]
    return PauliTensorSum(*tensors_to_sum)

def alpha_commutator(chain_length, J, h, lamb):
    single_terms = hamiltonian_single_terms(chain_length, J, h)
    int_terms = hamiltonian_interaction_terms(chain_length, lamb)
    int_int_single_commutator_norm = commutator(int_terms, commutator(int_terms, single_terms)).norm_upper_bound()
    single_single_int_commutator_norm  = commutator(single_terms, commutator(single_terms, int_terms)).norm_upper_bound()
    return int_int_single_commutator_norm/12 + single_single_int_commutator_norm/24

def trotter_layer_error(t, chain_length, J, h, lamb):
    return t**3*alpha_commutator(chain_length, J, h, lamb)

def trotter_error(t, layers, chain_length, J, h, lamb):
    return t**3/layers**2*alpha_commutator(chain_length, J, h, lamb)

def nlayers_trotter(t, epsilon, chain_length, J, h, lamb):
    return t**(3/2)/epsilon**(1/2)*alpha_commutator(chain_length, J, h, lamb)**(1/2)

def trotter_dt(epsilon, chain_length, J, h, lamb):
    return epsilon**(1/2)*alpha_commutator(chain_length, J, h, lamb)**(1/2)

#   ANALYTIC EXPRESSIONS
def nlayers_trotter_ana(t, epsilon, chain_length, J, h, lamb):
    return t*(4*np.abs(J**2*lamb)/3 + 2*np.abs(J*h*lamb)/3+(h**2*lamb)/3+np.abs(J*lamb**2)/3+np.abs(h*lamb**2)/6)*(chain_length-1)/epsilon**(1/3)

def error_trotter2_ana(t, nlayers, chain_length, J, h, lamb):
    return t**3/nlayers**3*(2/3*np.abs(J**2*lamb) + 2/3*np.abs(J*h*lamb) + 1/6*np.abs(h**2*lamb) + 2/3*np.abs(J*lamb**2) + 1/3*np.abs(h*lamb**2))*(chain_length - 1)

# --------------------------- 
#    CIRCUIT TRANSPILATION
# ---------------------------

def logical_string_quench_simulation_circuit(chain_length, J, h, lamb, dt, nlayers, g_func=None, particle_pair_left_position=None, particle_pair_length=None, x_basis=False, barriers=False, measure=True):
    if callable(g_func):
        g = g_func(chain_length, nlayers)
    elif g_func is not None:
        g = g_func
    else:
        g = None
    this_trotter_layers = []
    for k in range(nlayers):
        this_trotter_layers.append(SecondOrderTrotter(chain_length, J, h, lamb, dt, 1, g=g[:, k] if g is not None else g, x_basis=x_basis, barriers=barriers))
    if particle_pair_left_position is not None and particle_pair_length is not None:
        this_time_logical_circ = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    else:
        this_time_logical_circ = QuantumCircuit(2*chain_length-1)
    for this_trotter_layer in this_trotter_layers:
        this_time_logical_circ = this_time_logical_circ.compose(this_trotter_layer).decompose()
    if measure: this_time_logical_circ.measure_all()
    return this_time_logical_circ

def string_quench_simulation_circuits_dt(chain_length, J, h, lamb, final_time, dt, steps, backend, optimization_level, g_func=None, particle_pair_left_position=None, particle_pair_length=None, layout=None, x_basis=False, force_even=False, barriers=False, measure=True):
    t_arr = np.linspace(0, final_time, steps+1)
    nlayers_arr = (t_arr / dt).astype(int)[1:]
    nlayers_arr[np.equal(nlayers_arr, 0)] = 1
    if force_even:
        nlayers_arr[~np.equal(nlayers_arr % 2, 0)] +=   1
    dt_arr = t_arr[1:] / nlayers_arr
    trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    if particle_pair_left_position is not None and particle_pair_length is not None:
        state_preparation_logical_circ = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    else:
        state_preparation_logical_circ = QuantumCircuit(2*chain_length-1)
    measured_state_preparation_logical_circ = state_preparation_logical_circ.copy()
    if measure: measured_state_preparation_logical_circ.measure_all()
    logical_time_circs = [measured_state_preparation_logical_circ]
    with Pool(processes=cpu_count()) as pool:
        evolved_circuits = pool.starmap(logical_string_quench_simulation_circuit, [(chain_length, J, h, lamb, dt_arr[i], nlayers_arr[i], g_func, particle_pair_left_position, particle_pair_length, x_basis, barriers, measure) for i in range(steps)])
    # evolved_circuits = []
    # for i in range(steps):
    #     evolved_circuits.append(logical_string_quench_simulation_circuit(chain_length, J, h, lamb, dt_arr[i], nlayers_arr[i], g_func, particle_pair_left_position, particle_pair_length, x_basis, barriers, measure))
    logical_time_circs += evolved_circuits
    if layout is None:
        physical_time_circs = trotter_pm.run(logical_time_circs[:2])
        first_layout = physical_time_circs[1].layout.final_index_layout()
        trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=first_layout)
        physical_time_circs += trotter_pm.run(logical_time_circs[2:])
    else:
        physical_time_circs = trotter_pm.run(logical_time_circs)
    return physical_time_circs

def string_quench_simulation_circuit_mpf_single_t(chain_length, J, h, lamb, t, ks, pm, g_func=None, particle_pair_left_position=None, particle_pair_length=None, x_basis=False, barriers=False):
    logical_circs = []
    for k in ks:
        logical_circs.append(logical_string_quench_simulation_circuit(chain_length, J, h, lamb, t/k, k, g_func, particle_pair_left_position, particle_pair_length, x_basis))
    physical_circs = pm.run(logical_circs)
    return physical_circs

def string_quench_simulation_circuits_mpf(chain_length, J, h, lamb, dt, ks_matrix, backend, optimization_level, g_func=None, particle_pair_left_position=None, particle_pair_length=None, layout=None, x_basis=False, barriers=False, measure=True):
    ks_matrix = np.array(ks_matrix)
    trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    if particle_pair_left_position is not None and particle_pair_length is not None:
        state_preparation_logical_circ = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    else:
        state_preparation_logical_circ = QuantumCircuit(2*chain_length-1)
    measured_state_preparation_logical_circ = state_preparation_logical_circ.copy()
    if measure: measured_state_preparation_logical_circ.measure_all()
    measured_state_preparation_physical_circ = trotter_pm.run(measured_state_preparation_logical_circ)
    physical_time_circs = [[measured_state_preparation_physical_circ]*ks_matrix.shape[1]]
    t_arr = np.linspace(dt, dt*len(ks_matrix), len(ks_matrix))
    # with Pool(processes=cpu_count()) as pool:
    #     evolved_circuits = pool.starmap(string_quench_simulation_circuit_mpf_single_t, [(chain_length, J, h, lamb, t_arr[i], ks_matrix[i], trotter_pm, g_func, particle_pair_left_position, particle_pair_length, x_basis, force_even, barriers) for i in range(ks_matrix.shape[0])])
    evolved_circuits = []
    for i in range(ks_matrix.shape[0]):
        evolved_circuits.append(string_quench_simulation_circuit_mpf_single_t(chain_length, J, h, lamb, t_arr[i], ks_matrix[i], trotter_pm, g_func, particle_pair_left_position, particle_pair_length, x_basis, barriers))
    physical_time_circs += evolved_circuits
    return physical_time_circs

def logical_string_quench_odr_circuit(chain_length, J, h, lamb, dt, nlayers, g_func=None, particle_pair_left_position=None, particle_pair_length=None, x_basis=False, barriers=False):
    if callable(g_func):
        g = g_func(chain_length, nlayers)
    elif g_func is not None:
        g = g_func
    else:
        g = None
    if particle_pair_left_position is not None:
        state_preparation_logical_circ = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    else:
        state_preparation_logical_circ = QuantumCircuit(2*chain_length-1)
    this_forwards_trotter_layers = []
    this_backwards_trotter_layers = []
    for k in range(nlayers//2):
        this_forwards_trotter_layers.append(SecondOrderTrotter(chain_length, J, h, lamb, dt, 1, g=g[:, k] if g is not None else g, x_basis=x_basis, barriers=barriers))
        this_backwards_trotter_layers.append(this_forwards_trotter_layers[-1].inverse())
    this_time_logical_circ = state_preparation_logical_circ.copy()
    for forwards_trotter_layer in this_forwards_trotter_layers:
        this_time_logical_circ = this_time_logical_circ.compose(forwards_trotter_layer).decompose()
    this_time_logical_circ.barrier()
    for backwards_trotter_layer in reversed(this_backwards_trotter_layers):
        this_time_logical_circ = this_time_logical_circ.compose(backwards_trotter_layer).decompose()
    this_time_logical_circ.measure_all()
    return this_time_logical_circ

def string_quench_odr_compuncomp_circuit_mpf_single_t(chain_length, J, h, lamb, t, ks, pm, g_func=None, particle_pair_left_position=None, particle_pair_length=None, x_basis=False, barriers=False):
    ks[:-1][~np.equal(ks[:-1] % 2, 0)] -= 1
    if (ks[-1] % 2) != 0:
        ks[-1] += 1
    ks[np.equal(ks, 0)] = 2
    logical_circs = []
    for k in ks:
        logical_circs.append(logical_string_quench_odr_circuit(chain_length, J, h, lamb, t/k, k, g_func, particle_pair_left_position, particle_pair_length, x_basis))
    physical_circs = pm.run(logical_circs)
    return physical_circs

def string_quench_odr_compuncomp_circuits_dt(chain_length, J, h, lamb, final_time, dt, steps, backend, optimization_level, g=None, particle_pair_left_position=None, particle_pair_length=None, layout=None, x_basis=False, barriers=False):
    t_arr = np.linspace(0, final_time, steps+1)
    nlayers_arr = (t_arr / dt).astype(int)[1:]
    nlayers_arr[np.equal(nlayers_arr, 0)] = 1
    nlayers_arr[~np.equal(nlayers_arr % 2, 0)] += 1
    dt_arr = t_arr[1:] / nlayers_arr
    try:
        if (fractional_gates := backend.options.use_fractional_gates):
            sched_pm = PassManager([ALAPScheduleAnalysis(backend.instruction_durations), PadDelay(backend.instruction_durations)])
    except AttributeError:
        fractional_gates = False
    trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    if (particle_pair_left_position is not None) and (particle_pair_length is not None):
        state_preparation_logical_circ = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    else:
        state_preparation_logical_circ = QuantumCircuit(2*chain_length-1)
    state_preparation_logical_circ.measure_all()
    logical_time_circs = [state_preparation_logical_circ]
    with Pool(processes=cpu_count()) as pool:
        evolved_circuits = pool.starmap(logical_string_quench_odr_circuit, [(chain_length, J, h, lamb, dt_arr[i], nlayers_arr[i], g, particle_pair_left_position, particle_pair_length, x_basis, barriers) for i in range(steps)])
    evolved_circuits = []
    for i in range(steps):
        evolved_circuits.append(logical_string_quench_odr_circuit(chain_length, J, h, lamb, dt_arr[i], nlayers_arr[i], g, particle_pair_left_position, particle_pair_length, x_basis, barriers))
    logical_time_circs += evolved_circuits
    if layout is None:
        physical_time_circs = trotter_pm.run(logical_time_circs[:2])
        first_layout = physical_time_circs[1].layout.final_index_layout()
        trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=first_layout)
        physical_time_circs += trotter_pm.run(logical_time_circs[2:])
    else:
        physical_time_circs = trotter_pm.run(logical_time_circs)
    return physical_time_circs

def string_quench_odr_compuncomp_circuits_mpf(chain_length, J, h, lamb, dt, ks_matrix, backend, optimization_level, g_func=None, particle_pair_left_position=None, particle_pair_length=None, layout=None, x_basis=False, barriers=False, measure=True):
    ks_matrix = np.array(ks_matrix)
    trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    if (particle_pair_left_position is not None) and (particle_pair_length is not None):
        state_preparation_logical_circ = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    else:
        state_preparation_logical_circ = QuantumCircuit(2*chain_length-1)
    measured_state_preparation_logical_circ = state_preparation_logical_circ.copy()
    if measure: measured_state_preparation_logical_circ.measure_all()
    measured_state_preparation_physical_circ = trotter_pm.run(measured_state_preparation_logical_circ)
    physical_time_circs = [[measured_state_preparation_physical_circ]*ks_matrix.shape[1]]
    t_arr = np.linspace(dt, dt*len(ks_matrix), len(ks_matrix))
    # with Pool(processes=cpu_count()) as pool:
    #     evolved_circuits = pool.starmap(string_quench_odr_compuncomp_circuit_mpf_single_t, [(chain_length, J, h, lamb, t_arr[i], ks_matrix[i], trotter_pm, g_func, particle_pair_left_position, particle_pair_length, x_basis, barriers) for i in range(ks_matrix.shape[0])])
    evolved_circuits = []
    for i in range(ks_matrix.shape[0]):
        evolved_circuits.append(string_quench_odr_compuncomp_circuit_mpf_single_t(chain_length, J, h, lamb, t_arr[i], ks_matrix[i], trotter_pm, g_func, particle_pair_left_position, particle_pair_length, x_basis, barriers))
    physical_time_circs += evolved_circuits
    return physical_time_circs

def string_quench_simulation_and_odr_circs(chain_length, J, h, lamb, final_time, dt, steps, backend, optimization_level, g_func=None, particle_pair_left_position=None, particle_pair_length=None, layout=None, x_basis=False, barriers=False):
    seed = np.random.randint(100000)
    np.random.seed(seed)
    simulation_circuits = string_quench_simulation_circuits_dt(chain_length, J, h, lamb, final_time, dt, steps, backend, optimization_level, g_func, particle_pair_left_position, particle_pair_length, layout, x_basis, force_even=True, barriers=barriers)
    np.random.seed(seed)
    odr_circuits = string_quench_odr_compuncomp_circuits_dt(chain_length, J, h, lamb, final_time, dt, steps, backend, optimization_level, g_func, None, None, layout, x_basis, barriers)
    np.random.seed()
    return simulation_circuits, odr_circuits

def string_quench_simulation_and_odr_circs_mpf(chain_length, J, h, lamb, dt, ks_matrix, backend, optimization_level, g_func=None, particle_pair_left_position=None, particle_pair_length=None, layout=None, x_basis=False, barriers=False):
    seed = np.random.randint(100000)
    np.random.seed(seed)
    simulation_circuits = string_quench_simulation_circuits_mpf(chain_length, J, h, lamb, dt, ks_matrix, backend, optimization_level, g_func, particle_pair_left_position, particle_pair_length, layout, x_basis, barriers=barriers)
    np.random.seed(seed)
    odr_circuits = string_quench_odr_compuncomp_circuits_mpf(chain_length, J, h, lamb, dt, ks_matrix, backend, optimization_level, g_func, None, None, layout, x_basis, barriers)
    np.random.seed()
    return simulation_circuits, odr_circuits

def logical_singlep_quench_simulation_circuit(chain_length, J, h, lamb, particle_position, dt, nlayers, g_func=None, x_basis=False, barriers=False, measure=True):
    if callable(g_func):
        g = g_func(chain_length, nlayers)
    elif g_func is not None:
        g = g_func
    else:
        g = None
    this_trotter_layers = []
    for k in range(nlayers):
        this_trotter_layers.append(SecondOrderTrotter(chain_length, J, h, lamb, dt, 1, g=g[:, k] if g is not None else g, x_basis=x_basis, barriers=barriers))
    this_time_logical_circ = single_particle_initial_state(chain_length, particle_position, x_basis=x_basis)
    for this_trotter_layer in this_trotter_layers:
        this_time_logical_circ = this_time_logical_circ.compose(this_trotter_layer).decompose()
    if measure: this_time_logical_circ.measure_all()
    return this_time_logical_circ

def singlep_quench_simulation_circuits_dt(chain_length, J, h, lamb, particle_position, final_time, dt, steps, backend, optimization_level, g_func=None, layout=None, x_basis=False, force_even=False, barriers=False, measure=True):
    # g_func: func taking three arguments, chain_length, and dt, step and returning corresponding g arr
    t_arr = np.linspace(0, final_time, steps+1)
    nlayers_arr = (t_arr / dt).astype(int)[1:]
    nlayers_arr[np.equal(nlayers_arr, 0)] = 1
    if force_even: nlayers_arr[~np.equal(nlayers_arr % 2, 0)] += 1
    dt_arr = t_arr[1:] / nlayers_arr
    try:
        if (fractional_gates := backend.options.use_fractional_gates):
            sched_pm = PassManager([ALAPScheduleAnalysis(backend.instruction_durations), PadDelay(backend.instruction_durations)])
    except AttributeError:
        fractional_gates = False
    trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    state_preparation_logical_circ = single_particle_initial_state(chain_length, particle_position, x_basis=x_basis)
    measured_state_preparation_logical_circ = state_preparation_logical_circ.copy()
    if measure: measured_state_preparation_logical_circ.measure_all()
    logical_time_circs = [measured_state_preparation_logical_circ]
    # with Pool(processes=cpu_count()) as pool:
    #     evolved_circuits = pool.starmap(logical_singlep_quench_simulation_circuit, [(chain_length, J, h, lamb, particle_position, dt_arr[i], nlayers_arr[i], g_func, x_basis, barriers, measure) for i in range(steps)])
    evolved_circuits = []
    for i in range(steps):
        evolved_circuits.append(logical_singlep_quench_simulation_circuit(chain_length, J, h, lamb, particle_position, dt_arr[i], nlayers_arr[i], g_func, x_basis, barriers))
    logical_time_circs += evolved_circuits
    physical_time_circs = trotter_pm.run(logical_time_circs)
    if fractional_gates:
        physical_time_circs = sched_pm.run(physical_time_circs)
    return physical_time_circs

def singlep_quench_simulation_circuit_mpf_single_t(chain_length, J, h, lamb, t, ks, pm, g_func=None, particle_position=None, x_basis=False, barriers=False):
    logical_circs = []
    for k in ks:
        logical_circs.append(logical_singlep_quench_simulation_circuit(chain_length, J, h, lamb, particle_position, t/k, k, g_func, x_basis, barriers))
    physical_circs = pm.run(logical_circs)
    return physical_circs

def singlep_quench_simulation_circuits_mpf(chain_length, J, h, lamb, dt, ks_matrix, backend, optimization_level, g_func=None, particle_position=None, layout=None, x_basis=False, barriers=False):
    ks_matrix = np.array(ks_matrix)
    trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    if particle_position is not None:
        state_preparation_logical_circ = single_particle_initial_state(chain_length, particle_position, x_basis=x_basis)
    else:
        state_preparation_logical_circ = QuantumCircuit(2*chain_length-1)
    measured_state_preparation_logical_circ = state_preparation_logical_circ.copy()
    measured_state_preparation_logical_circ.measure_all()
    measured_state_preparation_physical_circ = trotter_pm.run(measured_state_preparation_logical_circ)
    physical_time_circs = [[measured_state_preparation_physical_circ] * ks_matrix.shape[1]]
    t_arr = np.linspace(dt, dt * len(ks_matrix), len(ks_matrix))
    evolved_circuits = []
    for i in range(ks_matrix.shape[0]):
        evolved_circuits.append(singlep_quench_simulation_circuit_mpf_single_t(chain_length, J, h, lamb, t_arr[i], ks_matrix[i], trotter_pm, g_func, particle_position, x_basis, barriers))
    physical_time_circs += evolved_circuits
    return physical_time_circs

def logical_singlep_quench_odr_compuncomp_circuit(chain_length, J, h, lamb, particle_position, dt, nlayers, g_func=None, x_basis=False, barriers=False, measure=True):
    if callable(g_func):
        g = g_func(chain_length, nlayers)
    elif g_func is not None:
        g = g_func
    else:
        g = None
    if particle_position is not None:
        state_preparation_logical_circ = single_particle_initial_state(chain_length, particle_position, x_basis=x_basis)
    else:
        state_preparation_logical_circ = QuantumCircuit(2*chain_length-1)
    this_forwards_trotter_layers = []
    this_backwards_trotter_layers = []
    for k in range(nlayers//2):
        this_forwards_trotter_layers.append(SecondOrderTrotter(chain_length, J, h, lamb, dt, 1, g=g[:, k] if g is not None else g, x_basis=x_basis, barriers=barriers))
        this_backwards_trotter_layers.append(this_forwards_trotter_layers[-1].inverse())
    this_time_logical_circ = state_preparation_logical_circ.copy()
    for forwards_trotter_layer in this_forwards_trotter_layers:
        this_time_logical_circ = this_time_logical_circ.compose(forwards_trotter_layer).decompose()
    this_time_logical_circ.barrier()
    for backwards_trotter_layer in reversed(this_backwards_trotter_layers):
        this_time_logical_circ = this_time_logical_circ.compose(backwards_trotter_layer).decompose()
    if measure: this_time_logical_circ.measure_all()
    return this_time_logical_circ

def closest_clifford_angle(angle):
    # clifford_angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    clifford_angles = np.array([np.pi/2, np.pi, 3*np.pi/2])
    min_ind = np.argmin(np.abs(((angle % (2*np.pi)) - clifford_angles)))
    return clifford_angles[min_ind]

def random_clifford_angle(angle):
    return np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2])

def logical_singlep_quench_odr_clifford_circuit(chain_length, J, h, lamb, particle_position, dt, nlayers, g_func=None, angle_func=closest_clifford_angle, x_basis=False, barriers=False, measure=True):
    if callable(g_func):
        g = g_func(chain_length, nlayers)
    elif g_func is not None:
        g = g_func
    else:
        g = None
    if g is not None:
        g = np.array([angle_func(gv) for gv in g.flatten()]).reshape(g.shape)
    J_clifford = closest_clifford_angle(J*dt)
    h_clifford = closest_clifford_angle(h*dt)
    lamb_clifford = closest_clifford_angle(lamb*dt)
    if particle_position is not None:
        state_preparation_logical_circ = single_particle_initial_state(chain_length, particle_position, x_basis=x_basis)
    else:
        state_preparation_logical_circ = QuantumCircuit(2*chain_length-1)
    this_trotter_layers = []
    for k in range(nlayers):
        this_trotter_layers.append(SecondOrderTrotter(chain_length, 2*J_clifford, 2*h_clifford, lamb_clifford, 1/2, 1, g=g[:, k] if g is not None else g, x_basis=x_basis, barriers=barriers))
    this_time_logical_circ = state_preparation_logical_circ.copy()
    for forwards_trotter_layer in this_trotter_layers:
        this_time_logical_circ = this_time_logical_circ.compose(forwards_trotter_layer).decompose()
    if measure: this_time_logical_circ.measure_all()
    return this_time_logical_circ

def logical_string_quench_odr_clifford_circuit(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, dt, nlayers, g_func=None, angle_func=closest_clifford_angle, x_basis=False, barriers=False, measure=True):
    if callable(g_func):
        g = g_func(chain_length, nlayers)
    elif g_func is not None:
        g = g_func
    else:
        g = None
    if g is not None:
        g = np.array([angle_func(gv) for gv in g.flatten()]).reshape(g.shape)
    J_clifford = closest_clifford_angle(J*dt)
    h_clifford = closest_clifford_angle(h*dt)
    lamb_clifford = closest_clifford_angle(lamb*dt)
    if particle_pair_left_position is not None and particle_pair_length is not None:
        state_preparation_logical_circ = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    else:
        state_preparation_logical_circ = QuantumCircuit(2*chain_length-1)
    this_trotter_layers = []
    for k in range(nlayers):
        this_trotter_layers.append(SecondOrderTrotter(chain_length, 2*J_clifford, 2*h_clifford, lamb_clifford, 1/2, 1, g=g[:, k] if g is not None else g, x_basis=x_basis, barriers=barriers))
    this_time_logical_circ = state_preparation_logical_circ.copy()
    for forwards_trotter_layer in this_trotter_layers:
        this_time_logical_circ = this_time_logical_circ.compose(forwards_trotter_layer).decompose()
    if measure: this_time_logical_circ.measure_all()
    return this_time_logical_circ

def singlep_quench_odr_compuncomp_circuits_dt(chain_length, J, h, lamb, final_time, dt, steps, backend, optimization_level, g_func=None, particle_position=None, layout=None, x_basis=False, barriers=False, measure=True):
    t_arr = np.linspace(0, final_time, steps+1)
    nlayers_arr = (t_arr / dt).astype(int)[1:]
    nlayers_arr[np.equal(nlayers_arr, 0)] = 1
    nlayers_arr[~np.equal(nlayers_arr % 2, 0)] += 1
    dt_arr = t_arr[1:] / nlayers_arr
    try:
        if (fractional_gates := backend.options.use_fractional_gates):
            sched_pm = PassManager([ALAPScheduleAnalysis(backend.instruction_durations), PadDelay(backend.instruction_durations)])
    except AttributeError:
        fractional_gates = False
    trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    if particle_position is not None:
        state_preparation_logical_circ = single_particle_initial_state(chain_length, particle_position, x_basis=x_basis)
    else:
        state_preparation_logical_circ = QuantumCircuit(2*chain_length-1)
    measured_state_preparation_logical_circ = state_preparation_logical_circ.copy()
    if measure: measured_state_preparation_logical_circ.measure_all()
    logical_time_circs = [measured_state_preparation_logical_circ]
    # with Pool(processes=cpu_count()) as pool:
    #     evolved_circuits = pool.starmap(logical_singlep_quench_odr_compuncomp_circuit, [(chain_length, J, h, lamb, particle_position, dt_arr[i], nlayers_arr[i], g_func, x_basis, barriers, measure) for i in range(steps)])
    evolved_circuits = []
    for i in range(steps):
        evolved_circuits.append(logical_singlep_quench_odr_compuncomp_circuit(chain_length, J, h, lamb, particle_position, dt_arr[i], nlayers_arr[i], g_func, x_basis, barriers))
    logical_time_circs += evolved_circuits
    physical_time_circs = trotter_pm.run(logical_time_circs)
    if fractional_gates:
        physical_time_circs = sched_pm.run(physical_time_circs)
    return physical_time_circs

def singlep_quench_odr_clifford_circuits_dt(chain_length, J, h, lamb, final_time, dt, steps, backend, optimization_level, g_func=None, particle_position=None, layout=None, x_basis=False, force_even=True, barriers=False, measure=True):
    t_arr = np.linspace(0, final_time, steps+1)
    nlayers_arr = (t_arr / dt).astype(int)[1:]
    nlayers_arr[np.equal(nlayers_arr, 0)] = 1
    if force_even: nlayers_arr[~np.equal(nlayers_arr % 2, 0)] += 1
    dt_arr = t_arr[1:] / nlayers_arr
    trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    if particle_position is not None:
        state_preparation_logical_circ = single_particle_initial_state(chain_length, particle_position, x_basis=x_basis)
    else:
        state_preparation_logical_circ = QuantumCircuit(2*chain_length-1)
    measured_state_preparation_logical_circ = state_preparation_logical_circ.copy()
    if measure: measured_state_preparation_logical_circ.measure_all()
    logical_time_circs = [measured_state_preparation_logical_circ]
    evolved_circuits = []
    for i in range(steps):
        evolved_circuits.append(logical_singlep_quench_odr_clifford_circuit(chain_length, J, h, lamb, particle_position, dt_arr[i], nlayers_arr[i], g_func, random_clifford_angle, x_basis, barriers))
    logical_time_circs += evolved_circuits
    physical_time_circs = trotter_pm.run(logical_time_circs)
    return physical_time_circs

def string_quench_odr_clifford_circuits_dt(chain_length, J, h, lamb, final_time, dt, steps, backend, optimization_level, g_func=None, particle_pair_left_position=None, particle_pair_length=None, layout=None, x_basis=False, force_even=True, barriers=False, measure=True):
    t_arr = np.linspace(0, final_time, steps+1)
    nlayers_arr = (t_arr / dt).astype(int)[1:]
    nlayers_arr[np.equal(nlayers_arr, 0)] = 1
    if force_even: nlayers_arr[~np.equal(nlayers_arr % 2, 0)] += 1
    dt_arr = t_arr[1:] / nlayers_arr
    trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    if particle_pair_left_position is not None and particle_pair_length is not None:
        state_preparation_logical_circ = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    else:
        state_preparation_logical_circ = QuantumCircuit(2*chain_length-1)
    measured_state_preparation_logical_circ = state_preparation_logical_circ.copy()
    if measure: measured_state_preparation_logical_circ.measure_all()
    logical_time_circs = [measured_state_preparation_logical_circ]
    evolved_circuits = []
    for i in range(steps):
        evolved_circuits.append(logical_string_quench_odr_clifford_circuit(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, dt_arr[i], nlayers_arr[i], g_func, random_clifford_angle, x_basis, barriers))
    logical_time_circs += evolved_circuits
    physical_time_circs = trotter_pm.run(logical_time_circs)
    return physical_time_circs

def singlep_quench_odr_compuncomp_circuit_mpf_single_t(chain_length, J, h, lamb, t, ks, pm, g_func=None, particle_position=None, x_basis=False, barriers=False):
    ks[:-1][~np.equal(ks[:-1] % 2, 0)] -= 1
    if (ks[-1] % 2) != 0:
        ks[-1] += 1
    ks[np.equal(ks, 0)] = 2
    logical_circs = []
    for k in ks:
        logical_circs.append(logical_singlep_quench_odr_compuncomp_circuit(chain_length, J, h, lamb, particle_position, t/k, k, g_func, x_basis, barriers))
    physical_circs = pm.run(logical_circs)
    return physical_circs

def singlep_quench_odr_compuncomp_circuits_mpf(chain_length, J, h, lamb, dt, ks_matrix, backend, optimization_level, g_func=None, particle_position=None, layout=None, x_basis=False, barriers=False):
    ks_matrix = np.array(ks_matrix)
    trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    if particle_position is not None:
        state_preparation_logical_circ = single_particle_initial_state(chain_length, particle_position, x_basis=x_basis)
    else:
        state_preparation_logical_circ = QuantumCircuit(2*chain_length-1)
    measured_state_preparation_logical_circ = state_preparation_logical_circ.copy()
    measured_state_preparation_logical_circ.measure_all()
    measured_state_preparation_physical_circ = trotter_pm.run(measured_state_preparation_logical_circ)
    physical_time_circs = [[measured_state_preparation_physical_circ] * ks_matrix.shape[1]]
    t_arr = np.linspace(dt, dt * len(ks_matrix), len(ks_matrix))
    evolved_circuits = []
    for i in range(ks_matrix.shape[0]):
        evolved_circuits.append(singlep_quench_odr_compuncomp_circuit_mpf_single_t(chain_length, J, h, lamb, t_arr[i], ks_matrix[i], trotter_pm, g_func, particle_position, x_basis, barriers))
    physical_time_circs += evolved_circuits
    return physical_time_circs

def singlep_quench_simulation_and_odr_compuncomp_circs(chain_length, J, h, lamb, particle_position, final_time, dt, steps, backend, optimization_level, g_func=None, layout=None, x_basis=False, barriers=False, measure=True):
    seed = np.random.randint(100000)
    np.random.seed(seed)
    simulation_circuits = singlep_quench_simulation_circuits_dt(chain_length, J, h, lamb, particle_position, final_time, dt, steps, backend, optimization_level, g_func, layout, x_basis, barriers=barriers, measure=measure)
    np.random.seed(seed)
    odr_circuits = singlep_quench_odr_compuncomp_circuits_dt(chain_length, J, h, lamb, final_time, dt, steps, backend, optimization_level, g_func, None, layout, x_basis, barriers, measure=measure)
    np.random.seed()
    return simulation_circuits, odr_circuits

def singlep_quench_simulation_and_odr_clifford_circs(chain_length, J, h, lamb, particle_position, final_time, dt, steps, backend, optimization_level, g_func=None, layout=None, x_basis=False, barriers=False, measure=True):
    seed = np.random.randint(100000)
    np.random.seed(seed)
    simulation_circuits = singlep_quench_simulation_circuits_dt(chain_length, J, h, lamb, particle_position, final_time, dt, steps, backend, optimization_level, g_func, layout, x_basis, force_even=True, barriers=barriers, measure=measure)
    np.random.seed(seed)
    odr_circuits = singlep_quench_odr_clifford_circuits_dt(chain_length, J, h, lamb, final_time, dt, steps, backend, 1, g_func, particle_position, layout, x_basis, force_even=True, barriers=barriers, measure=measure)
    np.random.seed(seed)
    aer_circuits = singlep_quench_odr_clifford_circuits_dt(chain_length, J, h, lamb, final_time, dt, steps, AerSimulator(method="stabilizer"), 1, g_func, particle_position, layout, x_basis, force_even=True, barriers=True)
    np.random.seed()
    return simulation_circuits, odr_circuits, aer_circuits

def string_quench_simulation_and_odr_clifford_circs(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, dt, steps, backend, optimization_level, g_func=None, layout=None, x_basis=False, barriers=False, measure=True):
    seed = np.random.randint(100000)
    np.random.seed(seed)
    simulation_circuits = string_quench_simulation_circuits_dt(chain_length, J, h, lamb, final_time, dt, steps, backend, optimization_level, g_func, particle_pair_left_position, particle_pair_length, layout, x_basis, force_even=True, barriers=barriers, measure=measure)
    np.random.seed(seed)
    odr_circuits = string_quench_odr_clifford_circuits_dt(chain_length, J, h, lamb, final_time, dt, steps, backend, 1, g_func, particle_pair_left_position, particle_pair_length, layout, x_basis, force_even=True, barriers=barriers, measure=measure)
    np.random.seed(seed)
    aer_circuits = string_quench_odr_clifford_circuits_dt(chain_length, J, h, lamb, final_time, dt, steps, AerSimulator(method="stabilizer"), 1, g_func, particle_pair_left_position, particle_pair_length, layout, x_basis, force_even=True, barriers=barriers, measure=measure)
    # aer_circuits = singlep_quench_odr_clifford_circuits_dt(chain_length, J, h, lamb, final_time, dt, steps, AerSimulator(method="stabilizer"), 1, g_func, particle_pair_left_position, particle_pair_length, layout, x_basis, force_even=True, barriers=barriers, measure=measure)
    np.random.seed()
    return simulation_circuits, odr_circuits, aer_circuits

def singlep_quench_simulation_and_odr_circs_mpf(chain_length, J, h, lamb, dt, ks_matrix, backend, optimization_level, g_func=None, particle_position=None, layout=None, x_basis=False, barriers=False):
    seed = np.random.randint(100000)
    np.random.seed(seed)
    simulation_circuits = singlep_quench_simulation_circuits_mpf(chain_length, J, h, lamb, dt, ks_matrix, backend, optimization_level, g_func, particle_position, layout, x_basis, barriers=barriers)
    np.random.seed(seed)
    odr_circuits = singlep_quench_odr_compuncomp_circuits_mpf(chain_length, J, h, lamb, dt, ks_matrix, backend, optimization_level, g_func, None, layout, x_basis, barriers)
    np.random.seed()
    return simulation_circuits, odr_circuits