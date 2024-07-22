from datetime import datetime
import utils.hexec as hexec
import json
import os

class benchmarkdb():
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

    def execute(self, nqubits_arr, depths_arr, backends_arr, physical_circuit_generating_func, observable_generating_fun, shots=4096, test_circuit_name=None, observable_name=None):
        # Create the circuits to run
        nqubits_arr = set(nqubits_arr)
        depths_arr = set(depths_arr)
        circuits = []
        for nqubits in nqubits_arr:
            for depth in depths_arr:
                circuits.append(physical_circuit_generating_func(nqubits, depth))

        # Populate database with general information
        thisid = 0 if len(self._data) == 0 else self._data[-1]["id"] + 1
        data_to_add = {"id": thisid, "nqubits_arr": sorted(list(nqubits_arr)), "depth_arr": sorted(list(depths_arr))}
        data_to_add["backends"] = sorted([backend.name for backend in backends_arr])
        data_to_add["test_circuit_name"] = test_circuit_name if test_circuit_name is not None else physical_circuit_generating_func.__name__
        data_to_add["observable_func_name"] = observable_name if observable_name is not None else observable_generating_fun.__name__
        data_to_add["jobs"] = {}

        # Run the jobs and populate job-specific available information
        estimator_opt_dict = {
            "default_shots": shots,
            "optimization_level": 0,
            "resilience_level": 0
        }
        for backend in backends_arr:
            this_jobs_arr = hexec.execute_estimator_batch(backend, estimator_opt_dict, circuits, observable_generating_fun)
            this_backend_list = [{
                "job_id": job.job_id(),
                "nqubits": circuit.num_qubits,
                "depth": circuit.depth(),
                "shots": shots,
                "creation_time": datetime.fromisoformat(job.metrics()["timestamps"]["created"]),
                "execution_time": None,
                "evs": []
            } for job, circuit in zip(this_jobs_arr, circuits)]
            data_to_add["jobs"][backend.name] = this_backend_list

        self._data.append(data_to_add)
        self.save()

    def update_status(self, service, check_all=False):
        for test in self._data[::-1]:
            one_to_be_completed = False
            for job_dicts in test["jobs"].values():
                for job_dict in job_dicts:
                    if job_dict["execution_time"] is None:
                        one_to_be_completed = True
                        job = service.job(job_dict["job_id"])
                        if job.in_final_state():
                            job_dict["execution_time"] = datetime.fromisoformat(job.metric()["timestamps"]["finished"])
                            job_dicts["evs"] = list(job.results()[0].data.evs)
            if (not one_to_be_completed) and (not check_all):
                break
        self.save()

    def _search_index_by_params(self, nqubits_arr=None, depths_arr=None, backends_arr=None, test_circuit_name=False, observable_name=None, limit=None):
        backends_name_arr = []
        for backend in backends_arr:
            if type(backend) != str:
                backends_arr.append(backend.name)
            else:
                backends_arr.append(backend)
        
        search_functions = []
        if nqubits_arr is not None:
            search_functions.append(lambda test: test["nqubits_arr"] == sorted(nqubits_arr))
        if depths_arr is not None:
            search_functions.append(lambda test: test["depth_arr"] == sorted(depths_arr))
        if backends_name_arr is not None:
            search_functions.append(lambda test: test["backends"] == sorted(backends_name_arr))
        if test_circuit_name is not None:
            search_functions.append(lambda test: test["test_circuit_name"] == test_circuit_name)
        if observable_name is not None:
            search_functions.append(lambda test: test["observable_name"] == observable_name)

        indices_to_return = []
        for i, test in enumerate(self._data[::-1]):
            passed = all([search_function(test) for search_function in search_functions])
            if passed:
                indices_to_return.append(len(self._data) - 1 - i)
            if (limit is not None) and (len(indices_to_return) >= limit): break
        
        return indices_to_return
    
    def _search_index_by_id(self, id):
        for i, test in enumerate(self._data[::-1]):
            if test["id"] == id:
                return i
        raise ValueError(f"No test was found with id: {id}")
    
    def search_by_params(self, nqubits_arr=None, depths_arr=None, backends_arr=None, test_circuit_name=False, observable_name=None, limit=None):
        indices = self._search_index_by_params(nqubits_arr, depths_arr, backends_arr, test_circuit_name, observable_name, limit)
        return [self._data[i] for i in indices]
    
    def search_by_id(self, id):
        ind = self._search_batch_index_by_id(id)
        return self._data[ind]
    
    # TODO: Some plotting functions