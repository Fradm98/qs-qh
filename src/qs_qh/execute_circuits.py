import numpy as np
from numpy import pi
from numpy.random import default_rng

from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import EstimatorOptions
from qiskit_ibm_runtime import Batch, EstimatorV2
from qiskit_ibm_runtime import ibm_backend

from qs_qh.circuits import *
from qs_qh.utils import *


def construct_logical_circuit(NUM_QUBITS: int, DEPTH: int, type_circ: QuantumCircuit, assign_params: str, dagger: bool=False):
    # Base circuit
    if type_circ == "mbl":
        logical_circuit = MBLCircuit(NUM_QUBITS, DEPTH // 2)  # Halved depth

    if dagger:
        # Compute-uncompute construct (doubles the depth)
        inverse = logical_circuit.inverse()
        logical_circuit.barrier()
        logical_circuit.compose(inverse, inplace=True)

    if assign_params == "random":
        # Parameter values
        rng = default_rng(seed=0)
        parameter_values = rng.uniform(-pi, pi, size=logical_circuit.num_parameters)
    elif assign_params == "fixed":
        parameter_values = np.linspace(-pi, pi, num=logical_circuit.num_parameters)
    parameter_values[0] = 0.3  # Fix interaction strength (specific to MBL circuit)
    logical_circuit.assign_parameters(parameter_values, inplace=True)
    return logical_circuit

def construct_logical_Pauli_observable(NUM_QUBITS: int, pauli_weight: int, type_pauli: str):
    paulis = ['I'*i + type_pauli*pauli_weight + 'I'*(NUM_QUBITS-i-pauli_weight) for i in range(NUM_QUBITS)]
    coeffs = 1/len(paulis)
    logical_observable = SparsePauliOp(paulis, coeffs)
    return logical_observable

def construct_physical_circuit(logical_circuit: QuantumCircuit, backend: ibm_backend, opt_level: int=3):
    pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend)
    physical_circuit = pm.run(logical_circuit)
    return physical_circuit

def construct_physical_Pauli_observable(physical_circuit: QuantumCircuit, logical_observable: SparsePauliOp):
    physical_layout = physical_circuit.layout
    physical_observable = logical_observable.apply_layout(physical_layout)
    return physical_observable

def estimator_options(shots: int=4096, err_suppr: int=0, err_mitig: int=0, zne: bool=True):
    options = EstimatorOptions()
    options.default_shots = shots
    options.optimization_level = err_suppr  # Deactivate error suppression
    options.resilience_level = err_mitig  # Deactivate error mitigation
    options.resilience.zne_mitigation = zne  # Activate ZNE error mitigation only
    if err_mitig == 0:
        options.resilience.measure_mitigation = False
    else:
        options.resilience.measure_mitigation = True
    return options

def construct_batch_execution(physical_circuit: QuantumCircuit, physical_observable: SparsePauliOp, backend: ibm_backend, options: EstimatorOptions, stat_shot: int, zne: bool):    
    jobs_batch = {}
    with Batch(backend=backend) as batch:
        estimator = EstimatorV2(session=batch, options=options)
        pub = (physical_circuit, physical_observable)

        if zne:
            # Linear extrapolation
            extrapolator = 'linear'
            estimator.options.resilience.zne.extrapolator = extrapolator
            jobs_batch[extrapolator] = estimator.run([pub])

            # Quadratic extrapolation
            extrapolator = 'polynomial_degree_2'
            estimator.options.resilience.zne.extrapolator = extrapolator
            jobs_batch[extrapolator] = estimator.run([pub])

            # Exponential extrapolation
            extrapolator = 'exponential'
            estimator.options.resilience.zne.extrapolator = extrapolator
            jobs_batch[extrapolator] = estimator.run([pub])

        else:
            # no zne
            extrapolator = 'no_zne'
            jobs_batch[extrapolator] = estimator.run([pub])


    jb = []
    for extrapolator, job_batch in jobs_batch.items():
        dict_batch = {"job_stat_number":stat_shot, "job_id":job_batch.job_id(), "extrapolator": extrapolator, "mem": options.resilience.measure_mitigation}
        jb.append(dict_batch)
    return jb

def construct_batches_execution(LIST_QUBITS: list, LIST_DEPTHS: list, backend=None, stat_shots: int=100, shots: int=4096, err_suppr: int=0, err_mitig: int=0, zne: bool=True, pauli_weight: int=1, type_pauli: str="Z"):    
    if backend == None:
        backend = least_busy_backend()
    options = estimator_options(shots=shots, err_suppr=err_suppr, err_mitig=err_mitig, zne=zne)

    jobs_batches = []
    for NUM_QUBITS in LIST_QUBITS: 
        for DEPTH in LIST_DEPTHS:
            print(f"construct logical circuit...")
            logical_circuit = construct_logical_circuit(NUM_QUBITS=NUM_QUBITS, DEPTH=DEPTH, type_circ="mbl", assign_params="fixed", dagger=True)
            print(f"construct logical observable...") 
            logical_observable = construct_logical_Pauli_observable(NUM_QUBITS=NUM_QUBITS, pauli_weight=pauli_weight, type_pauli=type_pauli)
            print(f"construct physical circuit...")
            physical_circuit = construct_physical_circuit(logical_circuit=logical_circuit, backend=backend)
            print(f"construct physical observable...")
            physical_observable = construct_physical_Pauli_observable(physical_circuit=physical_circuit, logical_observable=logical_observable)

            for shot in range(stat_shots):
                print(f"n_qubits: {NUM_QUBITS}, depth: {DEPTH}, stat: {shot}")
                jb = construct_batch_execution(physical_circuit=physical_circuit, physical_observable=physical_observable, backend=backend, options=options, stat_shot=shot, zne=zne)            
                jobs_batch = {"n_qubits": NUM_QUBITS, "depth": DEPTH, "batch": jb}
                jobs_batches.append(jobs_batch)
    if err_mitig == 0:
        mem = False
    else:
        mem = True
    save_jobs(filename=f"logs/jobs_qubits_{LIST_QUBITS}_depths_{LIST_DEPTHS}_pauli-{type_pauli}_weight_{pauli_weight}_stat_shots_{stat_shots}_zne_{zne}_mem_{mem}", jobs=jobs_batches)