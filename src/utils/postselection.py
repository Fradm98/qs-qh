from qiskit.quantum_info import SparsePauliOp, Statevector
import numpy as np

def measure_z_diagonal_post_selected_observables(sampler_jobs_objs, observable_generating_funcs, postselect_generating_func):    
    # TODO: This works for the Z basis, the X basis is the same but one has to add a layer of Hadamards.
    #       Doing the general case is too involved and not computationally efficient.
    if type(observable_generating_funcs) != list:
        observable_generating_funcs = [observable_generating_funcs]

    job_counts = []
    for job in sampler_jobs_objs:
        if job.in_final_state():
            job_counts.append(list(job.result()[0].data.values())[0].get_counts())
        else:
            raise TimeoutError("Some jobs are still to be finished")
    
    nqubits = len(list(job_counts[1].keys())[0])
    logical_observables = [observable_func(nqubits) for observable_func in observable_generating_funcs]
    logical_postselect_ops = postselect_generating_func(nqubits)
    
    # Commutation check
    print("Commutation check")
    logical_observables_sparseops = [SparsePauliOp(logical_observable) for logical_observable in logical_observables]
    logical_postselect_sparseops = [SparsePauliOp(logical_postselect_op) for logical_postselect_op in logical_postselect_ops]
    for i in range(len(logical_postselect_ops) - 1):
        left_paulis = logical_postselect_sparseops[i].simplify().paulis
        for j in range(i+1, len(logical_postselect_ops)):
            right_paulis = logical_postselect_sparseops[j].simplify().paulis
            commutes = True
            for pauli in left_paulis:
                commutes *= np.all(pauli.commutes(right_paulis))
            if not commutes:
                raise ValueError("All postselection operators must commute among themselves")
    for i in range(len(logical_postselect_ops) - 1):
        left_paulis = logical_postselect_sparseops[i].simplify().paulis
        for j in range(i+1, len(logical_observables_sparseops)):
            right_paulis = logical_observables_sparseops[j].simplify().paulis
            commutes = True
            for pauli in left_paulis:
                commutes *= np.all(pauli.commutes(right_paulis))
            if not commutes:
                raise ValueError("All observables must commute with the postselection operators")
        
    # Delete unphysical states and compute expectation values over physical basis states
    print("Computing observables")
    physical_job_counts = []
    expectation_values_cache = {}
    for counts_dict in job_counts:
        physical_counts_dict = {}
        for state_string, state_counts in counts_dict.items():
            this_statevector = Statevector([int(c) for c in state_string])
            is_physical_state = all([np.isclose(this_statevector.expectation_value(postselect_op), 1) for postselect_op in logical_postselect_sparseops])
            if is_physical_state:
                physical_counts_dict[state_string] = state_counts
                if state_string not in expectation_values_cache.keys():
                    expectation_values_cache[state_string] = [this_statevector.expectation_value(observable) for observable in logical_observables_sparseops]
        physical_job_counts.append(physical_counts_dict)

    # Compute the final expectations averaging over the physical basis expectation values weighted with the counts
    print("Computing expectations")  
    circ_observable_array = np.zeros((len(physical_job_counts), len(logical_observables)))
    for i, physical_counts_dict in enumerate(physical_job_counts):
        total_physical_counts = np.sum([counts for counts in physical_counts_dict.values()])
        for j in range(len(logical_observables)):
            circ_observable_array[i, j] = np.sum([counts*expectation_values_cache[basis_state][j]/total_physical_counts for basis_state, counts in physical_counts_dict.items()])

    return circ_observable_array

def compute_state_string_diagonal_expectation(z_basis_state_string, sparse_pauli_op):
    if len(sparse_pauli_op.group_commuting()) > 1:
        raise ValueError("All the components of sparse_pauli_op must commute")
    pauli_string_expectations = np.zeros(len(sparse_pauli_op.paulis), dtype=complex)
    state_arr = np.array([int(s) for s in z_basis_state_string], dtype=int)
    for i, pauli in enumerate(sparse_pauli_op.paulis):
        prefactor = 1+0j
        for qubit_state, pauli_str in zip(state_arr, str(pauli)):
            if pauli_str == "Y":
                prefactor *= -1j if qubit_state == 0 else 1j
        pauli_z_array = np.array([1 if pauli_str in ["X", "Y", "Z"] else 0 for pauli_str in str(pauli)], dtype=int)
        z_basis_expectation = (-1)**(np.sum(state_arr*pauli_z_array) % 2)
        pauli_string_expectations[i] = prefactor*z_basis_expectation
    return np.sum(sparse_pauli_op.coeffs*pauli_string_expectations)

def z_basis_pauli_string_prefactor(pauli_string):
    prefactor = 1+0j
    for pauli in pauli_string:
        if pauli == "Y":
            pass