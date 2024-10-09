# -----------------------------------------------
#           TOOLS FOR CIRCUIT EXECUTION
#                 IN HARDWARE
# -----------------------------------------------

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
from qiskit_ibm_runtime import Batch, EstimatorV2, SamplerV2
from utils.circs import check_and_measure_active_qubits
from itertools import product
import numpy as np
import json
import os

class ExecDB:
    """Database for saving jobs"""
    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            with open(path, "r") as f:
                self._data = json.load(f)
        else:
            self._data = []
            with open(path, "w") as f:
                json.dump(self._data, f, indent=4)   

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=4)

    def _search_batches_indices_by_params(self, batch_args, physical_circuits, observable_func_name, strict_depth=True, limit=10):
        indices_to_return = []
        observable_func_name = observable_func_name.__name__ if callable(observable_func_name) else observable_func_name
        for i, batch in enumerate(self._data[::-1]):
            is_equal = ([
                batch.get(key, None) == val for key, val in batch_args.items()
            ]
            +
            [
                np.array_equal(batch["nqubits_arr"], sorted(list({len(physical_circuit.layout.final_index_layout()) for physical_circuit in physical_circuits}))),
                np.array_equal(batch["depths_arr"], sorted(list({physical_circuit.depth() for physical_circuit in physical_circuits}))),
                batch["observables_func_name"] == observable_func_name
            ])
            if not strict_depth: del is_equal[-2]
            if all(is_equal): indices_to_return.append(len(self._data) - 1 - i)
            if len(indices_to_return) >= limit: break
        return indices_to_return
    
    def _search_batch_index_by_id(self, id):
        for i, batch in enumerate(self._data):
            if batch["id"] == id:
                return i
        raise ValueError(f"No batch found with id: {id}")

    def search_by_params(self, batch_args, physical_circuits, observable_func_name, strict_depth=True, ibmq_service=None, limit=10):
        indices = self._search_batches_indices_by_params(batch_args, physical_circuits, observable_func_name, strict_depth, limit)
        batches_to_return = [self._data[i] for i in indices]
        if ibmq_service is None:
            return batches_to_return[0] if len(batches_to_return) == 1 else batches_to_return
        else:
            jobs_objs = []
            for batch in batches_to_return:
                this_jobs_ids = [job["job_id"] for job in batch["jobs"]]
                this_jobs_objs = [ibmq_service.job(job_id=job_id) for job_id in this_jobs_ids]
                jobs_objs.append(this_jobs_objs)
            return jobs_objs[0] if len(batches_to_return) == 1 else jobs_objs

    def search_by_id(self, id):
        ind = self._search_batch_index_by_id(id)
        return self._data[ind]
    
    def add(self, batch_args, physical_circuits, observable_generating_func_name, job_ids_arr):
        thisid = 0 if len(self._data) == 0 else self._data[-1]["id"] + 1
        data_to_add = {"id": thisid}
        data_to_add = data_to_add | batch_args
        data_to_add["nqubits_arr"] = sorted(list({len(physical_circuit.layout.final_index_layout()) for physical_circuit in physical_circuits}))
        data_to_add["depths_arr"] = sorted(list({physical_circuit.depth() for physical_circuit in physical_circuits}))
        if callable(observable_generating_func_name):
            data_to_add["observables_func_name"] = observable_generating_func_name.__name__
        else:
            data_to_add["observables_func_name"] = str(observable_generating_func_name)
        data_to_add["jobs"] = [{"job_id": jid, "nqubits": len(physical_circuit.layout.final_index_layout()), "depth": physical_circuit.depth(), "layout": physical_circuit.layout.final_index_layout()} for jid, physical_circuit in zip(job_ids_arr, physical_circuits)]
        self._data.append(data_to_add)
        self.save()

    def remove(self, id):
        index = self._search_batch_index_by_id(id)
        self._data.pop(index)
        self.save()

    def execute_estimator_batch(self, backend, estimator_opt_dict, physical_circuits, observable_generating_func, observable_name=None):
        execute_estimator_batch(backend, estimator_opt_dict, physical_circuits, observable_generating_func, self, observable_name)

    def execute_sampler_batch(self, backend, sampler_opt_dict, physical_circuits):
        execute_sampler_batch(backend, sampler_opt_dict, physical_circuits, self)

def transpile(logical_circuits, optimization_level, backend, largest_layout=None):
    try:
        logical_circuits = list(logical_circuits)
    except TypeError:
        logical_circuits = [logical_circuits]

    nqubits_arr = [logical_circuit.num_qubits for logical_circuit in logical_circuits]
    arg_sort = np.argsort(nqubits_arr, kind="stable")
    logical_circuits = [logical_circuits[i] for i in arg_sort]
    nqubits_arr, counts = np.unique(nqubits_arr, return_counts=True)
    physical_circuits = []

    for i, nqubits in enumerate(nqubits_arr):
        pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=largest_layout[:nqubits] if largest_layout is not None else None)
        circuits_slice = slice(np.sum(counts[:i]), np.sum(counts[:i]) + counts[i])
        physical_circuits += pm.run(logical_circuits[circuits_slice])
    if len(physical_circuits) > 1:
        return physical_circuits
    else:
        return physical_circuits[0]
    
def map_obs_to_circs(transpiled_circuits, observable_generating_funcs, return_layouts=False):
    try:
        observable_generating_funcs = list(observable_generating_funcs)
    except TypeError:
        observable_generating_funcs = [observable_generating_funcs]
    
    if type(transpiled_circuits) != list:
        transpiled_circuits = [transpiled_circuits]

    mapped_observables = []
    if return_layouts: layouts = []
    for transpiled_circuit in transpiled_circuits:
        if transpiled_circuit.layout is not None:
            layout = transpiled_circuit.layout.final_index_layout()
            this_mapped_observables = []
            for observable_generating_func in observable_generating_funcs:
                logical_observable = SparsePauliOp(observable_generating_func(len(layout)))
                this_mapped_observables.append(logical_observable.apply_layout(transpiled_circuit.layout))
            if return_layouts: layouts.append(layout)
        else:
            this_mapped_observables = [observable_generating_func(transpiled_circuit.num_qubits) for observable_generating_func in observable_generating_funcs]
            if return_layouts: layouts.append(list(range(transpiled_circuit.num_qubits)))
        mapped_observables.append(this_mapped_observables)
    if return_layouts:
        return mapped_observables, layout
    else:
        return mapped_observables

def execute_estimator_batch(backend, estimator_opt_dict, transpiled_circuits, observable_generating_funcs, extra_options=None, job_db=None, observable_name=None):    
    if type(transpiled_circuits) != list:
        transpiled_circuits = [transpiled_circuits]

    mapped_observables =  map_obs_to_circs(transpiled_circuits, observable_generating_funcs)
    job_objs = []

    with Batch(backend=backend) as batch:
        estimator = EstimatorV2(mode=batch, options=estimator_opt_dict)
        for circ, obs in zip(transpiled_circuits, mapped_observables):
            pub = (circ, obs)
            job_objs.append(estimator.run([pub]))
    
    if job_db is not None:
        observables_func_name = observable_generating_funcs.__name__ if observable_name is None else observable_name
        job_ids = [job.job_id() for job in job_objs]
        if extra_options is not None:
            batch_args = extra_options | estimator_opt_dict
        else:
            batch_args = estimator_opt_dict
        job_db.add(batch_args, transpiled_circuits, observables_func_name, job_ids)

    return job_objs

def execute_sampler_batch(backend, sampler_opt_dict, transpiled_circuits, job_db=None):
    job_objs = []

    with Batch(backend=backend) as batch:
        sampler = SamplerV2(mode=batch, options=sampler_opt_dict)
        for circ in transpiled_circuits:
            circ = check_and_measure_active_qubits(circ)
            job_objs.append(sampler.run([circ]))

    if job_db is not None:
        job_ids = [job.job_id() for job in job_objs]
        job_db.add(sampler_opt_dict, transpiled_circuits, "Sampler", job_ids)

    return job_objs

def get_backend_best_qubit_chain(backend, nqubits):
    qlists = backend.properties().general_qlists
    for qlist in qlists:
        if qlist["name"] == f"lf_{nqubits:d}":
            return qlist["qubits"]
        
def create_estimator_options(default_shots, optimization_levels, zne_extrapolator_noise_levels, measure_mitigations, dd_sequences, enable_twirling):
    if type(default_shots) != list:
        default_shots = [default_shots]
    else:
        default_shots = sorted(list(set(default_shots)))
    if type(optimization_levels) != list:
        optimization_levels = [optimization_levels]
    else:
        optimization_levels = sorted(list(set(optimization_levels)))
    if type(zne_extrapolator_noise_levels) != list:
        zne_extrapolator_noise_levels = [zne_extrapolator_noise_levels]
    else:
        unique_zne_extrapolator_noise_levels = []
        for noise_level in zne_extrapolator_noise_levels:
            if noise_level not in unique_zne_extrapolator_noise_levels:
                unique_zne_extrapolator_noise_levels.append(noise_level)
        zne_extrapolator_noise_levels = unique_zne_extrapolator_noise_levels
    if type(measure_mitigations) != list:
        measure_mitigations = [measure_mitigations]
    else:
        measure_mitigations = list(set(measure_mitigations))
    if type(dd_sequences) != list:
        dd_sequences = [dd_sequences]
    else:
        dd_sequences = list(set(dd_sequences))
    if type(enable_twirling) != list:
        enable_twirling = [enable_twirling]
    else:
        enable_twirling = list(set(enable_twirling))
    
    estimator_options = []
    for shots, optimization_level, zne_extrapolator_noise_level, measure_mitigation, dd_sequence, twirling in product(default_shots, optimization_levels, zne_extrapolator_noise_levels, measure_mitigations, dd_sequences, enable_twirling):
        this_estimator_options = {
            "default_shots": shots,
            "optimization_level": optimization_level,
            "resilience_level": 0,
            "resilience": {
                "zne_mitigation": bool(zne_extrapolator_noise_level),
                "measure_mitigation": measure_mitigation,
                "pec_mitigation": False,
                "zne": {
                    "extrapolator": zne_extrapolator_noise_level[0], # zne_mitigation
                    "noise_factors": zne_extrapolator_noise_level[1]
                } if zne_extrapolator_noise_level else {}
            },
            "dynamical_decoupling": {
                "enable": bool(dd_sequence),
                "sequence_type": dd_sequence
            },
            "twirling": {
                "enable_gates": bool(twirling),
                "enable_measure": bool(twirling),
                "num_randomizations": "auto",
                "shots_per_randomization": "auto"
            }
        }
        estimator_options.append(this_estimator_options)
    return estimator_options