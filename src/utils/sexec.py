# -----------------------------------------------
#           TOOLS FOR CIRCUIT EXECUTION
#                IN SIMULATORS
# -----------------------------------------------

from qiskit_aer.primitives import EstimatorV2, SamplerV2
from utils.circs import check_and_measure_active_qubits

def execute_simulation_estimator_batch(simulator_options_dict, estimator_options_dict, circuits, observable_generating_funcs):
    job_objs = []
    
    estimator_options_dict["backend_options"] = simulator_options_dict

    estimator = EstimatorV2(options=estimator_options_dict)
    for circuit in circuits:
        observables = [observable_generating_func(circuit.num_qubits) for observable_generating_func in observable_generating_funcs]
        pub = (circuit, observables)
        job_objs.append(estimator.run([pub]))
    
    return job_objs

def execute_simulation_sampler_batch(simulator_options_dict, sampler_options_dict, circuits):
    job_objs = []

    options = {"backend_options": simulator_options_dict, "run_options": sampler_options_dict}
    
    sampler = SamplerV2(options=options, default_shots=sampler_options_dict.get("shots", 1024))
    for circuit in circuits:
        circuit = check_and_measure_active_qubits(circuit)
        job_objs.append(sampler.run([circuit]))

    return job_objs