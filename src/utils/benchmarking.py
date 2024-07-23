from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import matplotlib.dates as pltdates
import matplotlib.pyplot as plt
from itertools import product
from datetime import datetime
import utils.hexec as hexec
import utils.sexec as sexec
import numpy as np
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

    def execute(self, nqubits_arr, depths_arr, backends_arr, logical_circuit_generating_func, observable_generating_funcs, shots=4096, test_circuit_name=None, observable_name=None):
        nqubits_arr = check_and_convert_to_unique_list(nqubits_arr)
        depths_arr = check_and_convert_to_unique_list(depths_arr)
        try:
            backends_arr = list(backends_arr)
        except TypeError:
            backends_arr = [backends_arr]
        
        # Create the circuits to run
        circuits = []
        for nqubits in nqubits_arr:
            for depth in depths_arr:
                circuits.append(logical_circuit_generating_func(nqubits, depth))

        # Populate database with general information
        thisid = 0 if len(self._data) == 0 else self._data[-1]["id"] + 1
        data_to_add = {"id": thisid, "nqubits_arr": sorted(list(nqubits_arr)), "depth_arr": sorted(list(depths_arr))}
        data_to_add["backends"] = sorted([backend.name for backend in backends_arr])
        data_to_add["test_circuit_name"] = test_circuit_name if test_circuit_name is not None else logical_circuit_generating_func.__name__
        data_to_add["observable_func_name"] = observable_name if observable_name is not None else observable_generating_funcs.__name__
        data_to_add["jobs"] = {}

        # Run the jobs and populate job-specific available information
        estimator_opt_dict = {
            "default_shots": shots,
            "optimization_level": 0,
            "resilience_level": 0
        }
        for backend in backends_arr:
            pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
            physical_circuits = pm.run(circuits)
            this_jobs_arr = hexec.execute_estimator_batch(backend, estimator_opt_dict, physical_circuits, observable_generating_funcs)
            this_backend_list = [{
                "job_id": job.job_id(),
                "nqubits": circuit.num_qubits,
                "depth": circuit.depth(),
                "shots": shots,
                "creation_time": job.metrics()["timestamps"]["created"],
                "execution_time": None,
                "ev": None
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
                            job_dict["execution_time"] = job.metric()["timestamps"]["finished"]
                            job_dicts["ev"] = float(job.results()[0].data.evs[0])
            if (not one_to_be_completed) and (not check_all):
                break
        self.save()

    def _backends_objs_to_names(self, backends_arr):
        try:
            backends_arr = list(backends_arr)
        except TypeError:
            backends_arr = [backends_arr]
        backends_name_arr = []
        for backend in backends_arr:
            if type(backend) != str:
                backends_arr.append(backend.name)
            else:
                backends_arr.append(backend)
        return backends_name_arr
    
    def _date_range_to_datetime_range(self, date_range):
        return [datetime.fromisoformat(datestr) if type(datestr)==str else datestr for datestr in date_range]

    def _search_index_by_params(self, nqubits_arr=None, depths_arr=None, backends_arr=None, test_circuit_name_func=False, observable_name_func=None, date_range=None, limit=None):        
        nqubits_arr = check_and_convert_to_unique_list(nqubits_arr)
        depths_arr = check_and_convert_to_unique_list(depths_arr)
        try:
            backends_arr = list(backends_arr)
        except TypeError:
            backends_arr = [backends_arr]

        search_functions = []
        if nqubits_arr is not None:
            search_functions.append(lambda test: test["nqubits_arr"] == sorted(nqubits_arr))
        if depths_arr is not None:
            search_functions.append(lambda test: test["depth_arr"] == sorted(depths_arr))
        if backends_name_arr is not None:
            backends_name_arr = self._backends_objs_to_names(backends_arr)
            search_functions.append(lambda test: test["backends"] == sorted(backends_name_arr))
        if test_circuit_name is not None:
            test_circuit_name = test_circuit_name_func.__name__ if callable(test_circuit_name_func) else test_circuit_name_func
            search_functions.append(lambda test: test["test_circuit_name"] == test_circuit_name)
        if observable_name is not None:
            observable_name = observable_name_func.__name__ if callable(observable_name_func) else observable_name_func
            search_functions.append(lambda test: test["observable_name"] == observable_name)
        if date_range is not None:
            date_range = self._date_range_to_datetime_range(date_range)
            def date_range_searchf(test):
                execution_dates = []
                for job_dicts_arr in test["jobs"].values():
                    for job_dict in job_dicts_arr:
                        if (exectime := job_dict["execution_time"]) is not None:
                            execution_dates.append(datetime.fromisoformat(exectime))
                max_exec_time = max(execution_dates)
                min_exec_time = min(execution_dates)
                return (date_range[0] <= min_exec_time) and (date_range[1] >= max_exec_time)
            search_functions.append(date_range_searchf)

        indices_to_return = []
        for i, test in enumerate(self._data[::-1]):
            passed = all([search_function(test) for search_function in search_functions])
            if passed:
                indices_to_return.append(len(self._data) - 1 - i)
            if (limit is not None) and (len(indices_to_return) >= limit): 
                break
        
        return indices_to_return
    
    def _search_index_by_id(self, id):
        for i, test in enumerate(self._data[::-1]):
            if test["id"] == id:
                return i
        raise ValueError(f"No test was found with id: {id}")
    
    def search_by_params(self, nqubits_arr=None, depths_arr=None, backends_arr=None, test_circuit_name_func=None, observable_name_func=None, date_range=None, limit=None):
        indices = self._search_index_by_params(nqubits_arr, depths_arr, backends_arr, test_circuit_name_func, observable_name_func, date_range, limit)
        return [self._data[i] for i in indices]
    
    def search_by_id(self, id):
        ind = self._search_batch_index_by_id(id)
        return self._data[ind]

    def plot_mean_error_by_date(self, nqubits_arr, depths_arr, backends_arr, logical_circuit_generating_func, observable_generating_func, date_range=None, shots=4096, simulator_max_bond_dimension=256, simulation_results_folder_path=None):
        self.update_status()

        # Create the circuits to run
        nqubits_arr = set(nqubits_arr)
        depths_arr = set(depths_arr)
        circuits = []
        for nqubits in nqubits_arr:
            for depth in depths_arr:
                circuits.append(logical_circuit_generating_func(nqubits, depth))

        # Simulate the circuits
        estimator_options = {"default_precision": 1/np.sqrt(shots)}
        simulator_options = {
            "method": "matrix_product_state",
            "matrix_product_state_max_bond_dimension": simulator_max_bond_dimension,
            "matrix_product_state_truncation_threshold": 1e-10
        }
        physical_circuit_name = logical_circuit_generating_func.__name__
        observable_name = observable_generating_func.__name__
        results_filename = f"benchmark_simresults_nqubits_{min(nqubits)}-{max(nqubits)}_depths_{min(depths_arr)}-{max(depths_arr)}_testcirc_{physical_circuit_name}_obsname_{observable_name}_bd_{simulator_max_bond_dimension}.txt"
        results_filepath = os.path.join("" if simulation_results_folder_path is None else simulation_results_folder_path, results_filename)
        if not os.path.isfile(results_filepath):
            simulated_jobs = sexec.execute_simulation_estimator_batch(simulator_options, estimator_options, circuits, observable_generating_func)
            simulated_evs = [job.result()[0].data.evs for job in simulated_jobs]
            if simulation_results_folder_path is not None:
                np.savetxt(results_filepath, simulated_evs)
        
        # Get observables for each number of qubits and depths
        backends_name_arr = self._backends_objs_to_names(backends_arr)
        found_benchmarks = self.search_by_params(nqubits_arr, depths_arr, backends_arr, logical_circuit_generating_func, observable_generating_func, date_range)
        measurement_dates = {x:{} for x in product(nqubits_arr, depths_arr)}
        measured_evs = {x:{} for x in product(nqubits_arr, depths_arr)}
        for nqubits in nqubits_arr:
            for depth in depths_arr:
                this_measurement_dates = {backend_name: [] for backend_name in backends_name_arr}
                this_measured_evs = {backend_name: [] for backend_name in backends_name_arr}
                for benchmark in found_benchmarks:
                    for backend_name, jobs_dicts in benchmark["jobs"]:
                        for job_dict in jobs_dicts:
                            if (job_dict["nqubits"] == nqubits) and (job_dict["depth"] == depth):
                                this_measurement_dates[backend_name].append(datetime.fromisoformat(job_dict["execution_time"]))
                                this_measured_evs[backend_name].append(job_dict["ev"])
                measurement_dates[(nqubits, depth)] = this_measurement_dates
                measured_evs[(nqubits, depth)] = this_measured_evs

        # Plot the results
        fig, axs = plt.subplots(nrows=len(nqubits_arr)*len(depths_arr), figsize=[10, 7*len(nqubits_arr)*len(depths_arr)])
        for i, nqubits in enumerate(nqubits_arr):
            for j, depth in enumerate(depths_arr):
                for backend_name in backends_name_arr:
                    errors = np.abs(np.array(measured_evs[(nqubits, depth)][backend_name]) - simulated_evs[i+j])
                    date_locator = pltdates.AutoDateLocator()
                    date_formatter = pltdates.ConciseDateFormatter(date_locator, formats=['%Y', '%d/%b', '%d/%b', '%H:%M', '%H:%M', '%S.%f'], offset_formats=['', '%Y', '%b/%Y', '%d-%b-%Y', '%d/%b/%Y', '%d/%b/%Y %H:%M'])
                    axs[i+j].plot(measurement_dates[(nqubits, depth)][backend_name], errors, "o-", linewidth=2, markersize=8, label=backend_name)
                    axs[i+j].xaxis.set_major_locator(date_locator)
                    axs[i+j].xaxis.set_major_formatter(date_formatter)
                    axs[i+j].tick_params(axis="x", rotation=45, ha="right")
                    axs[i+j].set_title(f"Qubits: {nqubits} / Depth: {depth}")
                    axs[i+j].legend()
                    axs[i+j].grid()

def check_and_convert_to_unique_list(arg):
    try:
        arg = sorted(list(set(arg)))
    except TypeError:
        arg = [arg]
    return arg