from qiskit.quantum_info import SparsePauliOp, Statevector
import numpy as np

def measure_post_selected_observables(sampler_jobs_objs, observable_generating_funcs, postselect_generating_func):    
    try:
        observable_generating_funcs = list(observable_generating_funcs)
    except TypeError:
        observable_generating_funcs = [observable_generating_funcs]

    job_counts = [list(job.result()[0].data.values())[0].get_counts() for job in sampler_jobs_objs]
    nqubits = len(list(job_counts.keys())[0])
    logical_observables = [observable_func(nqubits) for observable_func in observable_generating_funcs]
    logical_postselect_ops = postselect_generating_func(nqubits)

    # Commutation check
    logical_observables_sparseops = []
    for logical_observable in logical_observables:
        logical_observables_sparseops.append(SparsePauliOp(logical_observable))
    logical_postslect_sparseops = []
    for logical_postselect_op in logical_postselect_ops:
        logical_postslect_sparseops.append(SparsePauliOp(logical_postselect_op))
    for logical_observable_sparseops in logical_observables_sparseops:
        commutes_with_all = all([logical_observable_sparseops.commutes(logical_postselect_sparseop) for logical_postselect_sparseop in logical_postslect_sparseops])
        if not commutes_with_all:
            raise ValueError("All observables must commute with the postselection operators")
        
    # Delete unphysical states and compute expectation values over physical basis states
    physical_job_counts = []
    expectation_values_cache = {}
    for counts_dict in job_counts:
        physical_counts_dict = {}
        for state_string, state_counts in counts_dict.items():
            this_statevector = Statevector([int(c) for c in state_string])
            is_physical_state = all([np.isclose(this_statevector.expectation_value(postselect_op), 1) for postselect_op in logical_observable_sparseops])
            if is_physical_state:
                physical_counts_dict[state_string] = state_counts
                if state_string not in expectation_values_cache.keys():
                    expectation_values_cache[state_string] = [this_statevector.expectation_value(observable) for observable in logical_observable_sparseops]
        physical_job_counts.append(physical_counts_dict)

    # Compute the final expectations averaging over the physical basis expectation values weighted with the counts    
    circ_observable_array = np.zeros((len(physical_job_counts, len(logical_observables))))
    for i, physical_counts_dict in enumerate(physical_job_counts):
        total_physical_counts = np.sum([counts for counts in physical_counts_dict.values()])
        for j in range(len(logical_observables)):
            circ_observable_array[i, j] = np.sum([counts*expectation_values_cache[basis_state]/total_physical_counts for basis_state, counts in physical_counts_dict.items()])

    return circ_observable_array