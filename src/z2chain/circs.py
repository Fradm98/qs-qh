from qiskit.transpiler.passes import Optimize1qGates, SetLayout, ApplyLayout, FullAncillaAllocation, Optimize1qGatesDecomposition, Optimize1qGatesSimpleCommutation
from qiskit_ibm_runtime.transpiler.passes.scheduling import ALAPScheduleAnalysis, PadDelay
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager, StagedPassManager, Layout
from utils.paulis import PauliTensor, PauliTensorSum, commutator
from qiskit.circuit import QuantumCircuit, Parameter
from sympy.physics.paulialgebra import Pauli
from utils.circs import remove_idle_qwires
from sympy.core.numbers import One
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
    def __init__(self, chain_length, x_basis=False):
        nqubits = 2*chain_length - 1
        super().__init__(nqubits)
        t_tau = Parameter("t_τ")
        t_sigma = Parameter("t_σ")
        if x_basis:
            self.rx(-2*t_tau, np.arange(0, self.num_qubits, 2))
            self.rx(-2*t_sigma, np.arange(1, self.num_qubits, 2))
        else:
            self.rz(-2*t_tau, np.arange(0, self.num_qubits, 2))
            self.rz(-2*t_sigma, np.arange(1, self.num_qubits, 2))

class TotalInteractionPropagator(QuantumCircuit):
    def __init__(self, chain_length, x_basis=False):
        nqubits = 2*chain_length - 1
        first_cnots_trgt_qubits = np.arange(0, nqubits-1, 2)
        first_cnots_ctrl_qubits = np.arange(1, nqubits, 2)
        second_cnots_trgt_qubits = np.arange(2, nqubits, 2)
        t_lamb = Parameter("t_lamb")
        super().__init__(nqubits)
        if x_basis:
            self.cx(first_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.cx(second_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.rz(-2*t_lamb, np.arange(1, nqubits, 2))
            self.cx(second_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.cx(first_cnots_trgt_qubits, first_cnots_ctrl_qubits)
        else:
            self.cx(first_cnots_ctrl_qubits, first_cnots_trgt_qubits)
            self.cx(first_cnots_ctrl_qubits, second_cnots_trgt_qubits)
            self.rx(-2*t_lamb, np.arange(1, nqubits, 2))
            self.cx(first_cnots_ctrl_qubits, second_cnots_trgt_qubits)
            self.cx(first_cnots_ctrl_qubits, first_cnots_trgt_qubits)

class TotalInteractionPropagatorDD(QuantumCircuit):
    def __init__(self, chain_length, t_g, std_tg_factor=0.5, x_basis=False):
        nqubits = 2*chain_length - 1
        first_cnots_trgt_qubits = np.arange(0, nqubits-1, 2)
        first_cnots_ctrl_qubits = np.arange(1, nqubits, 2)
        second_cnots_trgt_qubits = np.arange(2, nqubits, 2)
        t_lamb = Parameter("t_lamb")
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
            self.rz(-2*t_g, np.arange(1, nqubits, 2))
            self.cx(second_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.cx(first_cnots_trgt_qubits, first_cnots_ctrl_qubits)
        else:
            self.cx(first_cnots_ctrl_qubits, first_cnots_trgt_qubits)
            self.cx(first_cnots_ctrl_qubits, second_cnots_trgt_qubits)
            for i, qind in enumerate(range(0, nqubits, 2)):
                self.rz(-2*t_g[i], qind)
            self.rx(-2*t_lamb, np.arange(1, nqubits, 2))
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
        total_interaction_propagator = TotalInteractionPropagator(chain_length, x_basis).decompose()
    else:
        total_interaction_propagator = TotalInteractionPropagatorDD(chain_length, t_layer*g, x_basis=x_basis)
    total_interaction_propagator.assign_parameters([lamb*t_layer], inplace=True)
    total_single_body_propagator = TotalSingleBodyPropagator(chain_length, x_basis)
    total_single_body_propagator.assign_parameters([h*t_layer/2, t_layer*J/2], inplace=True)
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

def trotter_error(t, chain_length, J, h, lamb):
    single_terms = hamiltonian_single_terms(chain_length, J, h)
    int_terms = hamiltonian_interaction_terms(chain_length, lamb)
    int_int_single_commutator_norm = commutator(int_terms, commutator(int_terms, single_terms)).norm_upper_bound()
    single_single_int_commutator_norm  = commutator(single_terms, commutator(single_terms, int_terms)).norm_upper_bound()
    return t**3/12*int_int_single_commutator_norm + t**3/24*single_single_int_commutator_norm

def nlayers_trotter(t, epsilon, chain_length, J, h, lamb):
    single_terms = hamiltonian_single_terms(chain_length, J, h)
    int_terms = hamiltonian_interaction_terms(chain_length, lamb)
    int_int_single_commutator_norm = commutator(int_terms, commutator(int_terms, single_terms)).norm_upper_bound()
    single_single_int_commutator_norm  = commutator(single_terms, commutator(single_terms, int_terms)).norm_upper_bound()
    return t/12/epsilon**(1/3)*int_int_single_commutator_norm + t/24/epsilon**(1/3)*single_single_int_commutator_norm

def trotter_dt(epsilon, chain_length, J, h, lamb):
    single_terms = hamiltonian_single_terms(chain_length, J, h)
    int_terms = hamiltonian_interaction_terms(chain_length, lamb)
    int_int_single_commutator_norm = commutator(int_terms, commutator(int_terms, single_terms)).norm_upper_bound()
    single_single_int_commutator_norm  = commutator(single_terms, commutator(single_terms, int_terms)).norm_upper_bound()
    return (epsilon/(int_int_single_commutator_norm/12 + single_single_int_commutator_norm/24))**(1/3)

#   ANALYTIC EXPRESSIONS

def nlayers_trotter_ana(t, epsilon, chain_length, J, h, lamb):
    return t*(4*np.abs(J**2*lamb)/3 + 2*np.abs(J*h*lamb)/3+(h**2*lamb)/3+np.abs(J*lamb**2)/3+np.abs(h*lamb**2)/6)*(chain_length-1)/epsilon**(1/3)

def error_trotter2_ana(t, nlayers, chain_length, J, h, lamb):
    return t**3/nlayers**3*(2/3*np.abs(J**2*lamb) + 2/3*np.abs(J*h*lamb) + 1/6*np.abs(h**2*lamb) + 2/3*np.abs(J*lamb**2) + 1/3*np.abs(h*lamb**2))*(chain_length - 1)

# --------------------------- 
#    CIRCUIT TRANSPILATION
# ---------------------------

def string_quench_simulation_circuits_dt(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, dt, steps, backend, optimization_level, g=None, layout=None, x_basis=False, force_even=False, barriers=False):
    t_arr = np.linspace(0, final_time, steps+1)
    nlayers_arr = (t_arr // dt + 1).astype(int)
    if force_even:
        nlayers_arr[~np.equal(nlayers_arr % 2, 0)] += 1
    dt_arr = t_arr / nlayers_arr
    try:
        if (fractional_gates := backend.options.use_fractional_gates):
            sched_pm = PassManager([ALAPScheduleAnalysis(backend.instruction_durations), PadDelay(backend.instruction_durations)])
    except AttributeError:
        fractional_gates = False
    trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    state_preparation_logical_circ = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    measured_state_preparation_logical_circ = state_preparation_logical_circ.copy()
    measured_state_preparation_logical_circ.measure_all()
    state_preparation_physical_circ = trotter_pm.run(measured_state_preparation_logical_circ)
    if fractional_gates:
        state_preparation_physical_circ = sched_pm.run(state_preparation_physical_circ)
    circs_to_return = [state_preparation_physical_circ]
    for i in range(steps):
        this_trotter_logical_layer = SecondOrderTrotter(chain_length, J, h, lamb, dt_arr[i+1], 1, g=J*dt_arr[i+1] if g is not None else g, x_basis=x_basis, barriers=barriers)
        this_time_logical_circ = state_preparation_logical_circ.compose(this_trotter_logical_layer.repeat(nlayers_arr[i+1])).decompose()
        this_time_logical_circ.measure_all()
        this_time_physical_circ = trotter_pm.run(this_time_logical_circ)
        if fractional_gates:
            this_time_physical_circ = sched_pm.run(this_time_physical_circ)
        circs_to_return.append(this_time_physical_circ)
    return circs_to_return

def string_quench_odr_compuncomp_circuits_dt(chain_length, J, h, lamb, final_time, dt, steps, backend, optimization_level, g=None, particle_pair_left_position=None, particle_pair_length=None, layout=None, x_basis=False, barriers=False):
    t_arr = np.linspace(0, final_time, steps+1)
    nlayers_arr = (t_arr // dt + 1).astype(int)
    nlayers_arr[~np.equal(nlayers_arr % 2, 0)] += 1
    dt_arr = t_arr / nlayers_arr
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
    first_circuit = QuantumCircuit(2*chain_length-1)
    measured_first_circuit = first_circuit.copy()
    measured_first_circuit.measure_all()
    state_preparation_physical_circ = trotter_pm.run(measured_first_circuit)
    if fractional_gates:
        state_preparation_physical_circ = sched_pm.run(state_preparation_physical_circ)
    circs_to_return = [state_preparation_physical_circ]
    for i in range(steps):
        this_forwards_trotter_logical_layer = SecondOrderTrotter(chain_length, J, h, lamb, dt_arr[i+1], 1, g=J*dt_arr[i+1] if g is not None else g, x_basis=x_basis, barriers=barriers)
        this_backwards_trotter_logical_layer = this_forwards_trotter_logical_layer.inverse()
        this_time_logical_circ = state_preparation_logical_circ.compose(this_forwards_trotter_logical_layer.repeat(nlayers_arr[i+1]//2)).decompose()
        this_time_logical_circ.barrier()
        this_time_logical_circ = this_time_logical_circ.compose(this_backwards_trotter_logical_layer.repeat(nlayers_arr[i+1]//2)).decompose()
        this_time_logical_circ = this_time_logical_circ.compose(state_preparation_logical_circ).decompose()
        this_time_logical_circ.measure_all()
        this_time_physical_circ = trotter_pm.run(this_time_logical_circ)
        if fractional_gates:
            this_time_physical_circ = sched_pm.run(this_time_physical_circ)
        circs_to_return.append(this_time_physical_circ)
    return circs_to_return

def string_quench_simulation_and_odr_circs(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, dt, steps, backend, optimization_level, g=None, layout=None, x_basis=False, barriers=False):
    seed = np.random.randint(100000)
    np.random.seed(seed)
    simulation_circuits = string_quench_simulation_circuits_dt(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, dt, steps, backend, optimization_level, g, layout, x_basis, force_even=True, barriers=barriers)
    np.random.seed(seed)
    odr_circuits = string_quench_odr_compuncomp_circuits_dt(chain_length, J, h, lamb, final_time, dt, steps, backend, optimization_level, g, None, None, layout, x_basis, barriers)
    np.random.seed()
    return simulation_circuits, odr_circuits

def singlep_quench_simulation_circuits_dt(chain_length, J, h, lamb, particle_position, final_time, dt, steps, backend, optimization_level, g_func=0, layout=None, x_basis=False, barriers=False):
    # g_func: func taking three arguments, chain_length, and steps, step and returning corresponding g arr
    t_arr = np.linspace(0, final_time, steps+1)
    nlayers_arr = (t_arr // dt + 1).astype(int)
    nlayers_arr[~np.equal(nlayers_arr % 2, 0)] += 1
    dt_arr = t_arr / nlayers_arr
    try:
        if (fractional_gates := backend.options.use_fractional_gates):
            sched_pm = PassManager([ALAPScheduleAnalysis(backend.instruction_durations), PadDelay(backend.instruction_durations)])
    except AttributeError:
        fractional_gates = False
    trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    state_preparation_logical_circ = single_particle_initial_state(chain_length, particle_position, x_basis=x_basis)
    measured_state_preparation_logical_circ = state_preparation_logical_circ.copy()
    measured_state_preparation_logical_circ.measure_all()
    state_preparation_physical_circ = trotter_pm.run(measured_state_preparation_logical_circ)
    if fractional_gates:
        state_preparation_physical_circ = sched_pm.run(state_preparation_physical_circ)
    circs_to_return = [state_preparation_physical_circ]
    logical_time_circs = [state_preparation_logical_circ]
    for i in range(steps):
        if callable(g_func):
            g = [g_func(chain_length, nlayers_arr[i+1], k) for k in range(nlayers_arr[i+1])]
        this_trotter_logical_layer = SecondOrderTrotter(chain_length, J, h, lamb, dt_arr[i+1], 1, g=g, x_basis=x_basis, barriers=barriers)
        this_time_logical_circ = logical_time_circs[-1].copy()
        this_time_logical_circ 
        # this_trotter_logical_layer = SecondOrderTrotter(chain_length, J, h, lamb, dt_arr[i+1], 1, g=g, x_basis=x_basis, barriers=barriers)
        this_time_logical_circ = state_preparation_logical_circ.compose(this_trotter_logical_layer.repeat(nlayers_arr[i+1])).decompose()
        this_time_logical_circ.measure_all()
        this_time_physical_circ = trotter_pm.run(this_time_logical_circ)
        if fractional_gates:
            this_time_physical_circ = sched_pm.run(this_time_physical_circ)
        circs_to_return.append(this_time_physical_circ)
    return circs_to_return

def singlep_quench_odr_compuncomp_circuits_dt(chain_length, J, h, lamb, final_time, dt, steps, backend, optimization_level, g=None, particle_position=None, layout=None, x_basis=False, barriers=False):
    t_arr = np.linspace(0, final_time, steps+1)
    nlayers_arr = (t_arr // dt + 1).astype(int)
    nlayers_arr[~np.equal(nlayers_arr % 2, 0)] += 1
    dt_arr = t_arr / nlayers_arr
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
    measured_state_preparation_logical_circ.measure_all()
    state_preparation_physical_circ = trotter_pm.run(measured_state_preparation_logical_circ)
    if fractional_gates:
        state_preparation_physical_circ = sched_pm.run(state_preparation_physical_circ)
    circs_to_return = [state_preparation_physical_circ]
    for i in range(steps):
        this_forwards_trotter_logical_layer = SecondOrderTrotter(chain_length, J, h, lamb, dt_arr[i+1], 1, g=J*dt_arr[i+1] if g is not None else g, x_basis=x_basis, barriers=barriers)
        # this_forwards_trotter_logical_layer = SecondOrderTrotter(chain_length, J, h, lamb, dt_arr[i+1], 1, g=g, x_basis=x_basis, barriers=barriers)
        this_backwards_trotter_logical_layer = this_forwards_trotter_logical_layer.inverse()
        this_time_logical_circ = state_preparation_logical_circ.compose(this_forwards_trotter_logical_layer.repeat(nlayers_arr[i+1]//2)).decompose()
        this_time_logical_circ.barrier()
        this_time_logical_circ = this_time_logical_circ.compose(this_backwards_trotter_logical_layer.repeat(nlayers_arr[i+1]//2)).decompose()
        this_time_logical_circ = this_time_logical_circ.compose(state_preparation_logical_circ).decompose()
        this_time_logical_circ.measure_all()
        this_time_physical_circ = trotter_pm.run(this_time_logical_circ)
        if fractional_gates:
            this_time_physical_circ = sched_pm.run(this_time_physical_circ)
        circs_to_return.append(this_time_physical_circ)
    return circs_to_return

def singlep_quench_simulation_and_odr_circs(chain_length, J, h, lamb, particle_position, final_time, dt, steps, backend, optimization_level, g=None, layout=None, x_basis=False, barriers=False):
    seed = np.random.randint(100000)
    np.random.seed(seed)
    simulation_circuits = singlep_quench_simulation_circuits_dt(chain_length, J, h, lamb, particle_position, final_time, dt, steps, backend, optimization_level, g, layout, x_basis, barriers=barriers)
    np.random.seed(seed)
    odr_circuits = singlep_quench_odr_compuncomp_circuits_dt(chain_length, J, h, lamb, final_time, dt, steps, backend, optimization_level, g, None, layout, x_basis, barriers)
    np.random.seed()
    return simulation_circuits, odr_circuits