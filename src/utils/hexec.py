# -----------------------------------------------
#           TOOLS FOR CIRCUIT EXECUTION
#                 IN HARDWARE
# -----------------------------------------------

from qiskit_ibm_runtime import Batch, EstimatorV2
import numpy as np
import json
import os

class execdb:
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

    def _search_batches_indices_by_params(self, batch_args, physical_circuits, observable_func_name, limit=10):
        indices_to_return = []
        observable_func_name = observable_func_name.__name__ if callable(observable_func_name) else observable_func_name
        for i, batch in enumerate(self._data[::-1]):
            is_equal = all([
                batch[key] == val for key, val in batch_args.items()
            ]
            +
            [
                np.array_equal(batch["nqubits_arr"], sorted(list({len(physical_circuit.layout.final_index_layout()) for physical_circuit in physical_circuits}))),
                np.array_equal(batch["depths_arr"], sorted(list({physical_circuit.depth() for physical_circuit in physical_circuits}))),
                batch["observables_func_name"] == observable_func_name
            ])
            if is_equal: indices_to_return.append(len(self._data) - 1 - i)
            if len(indices_to_return) > limit: break
        return indices_to_return
    
    def _search_batch_index_by_id(self, id):
        for i, batch in enumerate(self._data):
            if batch["id"] == id:
                return i
        raise ValueError(f"No batch found with id: {id}")

    def search_by_params(self, batch_args, physical_circuits, observable_func_name, limit=10):
        indices = self._search_batches_indices_by_params(batch_args, physical_circuits, observable_func_name, limit)
        batches_to_return = [self._data[i] for i in indices]
        return batches_to_return

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

def execute_estimator_batch(backend, estimator_opt_dict, physical_circuits, observable_generating_func, job_db=None, observable_name=None):    
    job_objs = []
    layouts = []
    
    with Batch(backend=backend) as batch:
        estimator = EstimatorV2(session=batch, options=estimator_opt_dict)
        for physical_circuit in physical_circuits:
            layout = physical_circuit.layout.final_index_layout()
            logical_observable = observable_generating_func(len(layout))
            physical_observable = logical_observable.apply_layout(physical_circuit.layout)
            pub = (physical_circuit, physical_observable)
            layouts.append(layout)
            job_objs.append(estimator.run([pub]))
    
    if job_db is not None:
        observables_func_name = observable_generating_func.__name__ if observable_name is None else observable_name
        job_ids = [job.job_id() for job in job_objs]
        job_db.add(estimator_opt_dict, physical_circuits, observables_func_name, job_ids)