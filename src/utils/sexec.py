# -----------------------------------------------
#           TOOLS FOR CIRCUIT EXECUTION
#                IN SIMULATORS
# -----------------------------------------------

from qiskit_aer.primitives import EstimatorV2

def execute_simulation_estimator_batch(simulator_options_dict, estimator_options_dict, circuits, observable_generating_funcs):
    job_objs = []
    
    estimator_options_dict["backend_options"] = simulator_options_dict

    estimator = EstimatorV2(options=estimator_options_dict)
    for circuit in circuits:
        observables = [observable_generating_func(circuit.num_qubits) for observable_generating_func in observable_generating_funcs]
        pub = (circuit, observables)
        job_objs.append(estimator.run([pub]))
    
    return job_objs