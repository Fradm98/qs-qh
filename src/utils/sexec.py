# -----------------------------------------------
#           TOOLS FOR CIRCUIT EXECUTION
#                IN SIMULATORS
# -----------------------------------------------

from qiskit_aer.primitives import EstimatorV2, SamplerV2
from utils.circs import check_and_measure_active_qubits
from utils.hexec import map_obs_to_circs

def execute_simulation_estimator_batch(simulator_options_dict, estimator_options_dict, circuits, observable_generating_funcs):
    if type(circuits) != list:
        circuits = [circuits]
    if type(observable_generating_funcs) != list:
        observable_generating_funcs = [observable_generating_funcs]

    job_objs = []
    
    estimator_options_dict["backend_options"] = simulator_options_dict

    mapped_observables =  map_obs_to_circs(circuits, observable_generating_funcs)

    estimator = EstimatorV2(options=estimator_options_dict)
    for circuit, obs in zip(circuits, mapped_observables):
        pub = (circuit, obs)
        job_objs.append(estimator.run([pub]))
    
    return job_objs

def execute_simulation_sampler_batch(simulator_options_dict, sampler_options_dict, circuits):
    if type(circuit) != list:
        circuits = [circuits]
    
    job_objs = []

    options = {"backend_options": simulator_options_dict, "run_options": sampler_options_dict}
    
    sampler = SamplerV2(options=options, default_shots=sampler_options_dict.get("shots", 1024))
    for circuit in circuits:
        circuit = check_and_measure_active_qubits(circuit)
        job_objs.append(sampler.run([circuit]))

    return job_objs