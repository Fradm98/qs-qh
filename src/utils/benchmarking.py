from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passes import RemoveBarriers
import matplotlib.dates as pltdates
import matplotlib.pyplot as plt
from itertools import product
from datetime import datetime
import utils.hexec as hexec
import utils.sexec as sexec
import numpy as np
import subprocess
import plistlib
import getpass
import atexit
import json
import sys
import os

class BenchmarkDB():
    def __init__(self, path, password=None):
        self.path = path
        self.changed_permissions = False
        self.password = getpass.getpass(prompt="Introduce superuser password") if password is None else password
        atexit.register(self.clean_permissions)
        if os.path.exists(path):
            with open(path, "r") as f:
                self._data = json.load(f)
        else:
            self._data = []
            with open(path, "w") as f:
                json.dump(self._data, f, indent=4)   

    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self._data, f, indent=4)
        except PermissionError:
            subprocess.run(["sudo", "-S", "chown", "cobos:wheel", self.path], input=f"{self.password}\n", text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.changed_permissions = True
            with open(self.path, "w") as f:
                json.dump(self._data, f, indent=4)
    
    def clean_permissions(self):
        self.save()
        if self.changed_permissions:
            subprocess.run(["sudo", "-S", "chown", "root:wheel", self.path], input=f"{self.password}\n", text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def execute(self, nqubits_arr, depths_arr, devices_arr, service, logical_circuit_generating_func, observable_generating_func, estimator_opt_dicts, test_circuit_name=None, observable_name=None, optimization_level=2):
        nqubits_arr = check_and_convert_to_unique_list(nqubits_arr)
        depths_arr = check_and_convert_to_unique_list(depths_arr)
        
        # Call the backends from the device list
        backends_arr = [service.backend(device) for device in devices_arr]
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
        data_to_add["observable_func_name"] = observable_name if observable_name is not None else observable_generating_func.__name__
        # data_to_add["estimator_opt_dict"] = estimator_opt_dict
        data_to_add["jobs"] = {}

        # Run the jobs and populate job-specific available information
        for backend in backends_arr:
            data_to_add["jobs"][backend.name] = []
            for estimator_opt_dict in estimator_opt_dicts:
                pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend)
                physical_circuits = pm.run(circuits)
                this_jobs_arr = hexec.execute_estimator_batch(backend, estimator_opt_dict, physical_circuits, observable_generating_func)
                this_backend_list = [{
                    "job_id": job.job_id(),
                    "nqubits": nqubits,
                    "depth": depth,
                    "creation_time": job.metrics()["timestamps"]["created"],
                    "execution_time": None,
                    "ev": None,
                    "estimator_opt_dict": estimator_opt_dict
                } for job, (nqubits, depth) in zip(this_jobs_arr, product(nqubits_arr, depths_arr))]
                data_to_add["jobs"][backend.name] += this_backend_list

        self._data.append(data_to_add)
        self.save()

    def update_status(self, service, check_all=False, print_mode=False):
        for tn, test in enumerate(self._data[::-1]):
            one_to_be_completed = False
            for jdn, job_dicts in enumerate(test["jobs"].values()):
                for jn, job_dict in enumerate(job_dicts):
                    if print_mode: print(f"\rUpdating benchmark index: {len(self._data) - tn}/{len(self._data)}, Backend: {jdn + 1}/{len(test["jobs"].values())}, Job: {jn + 1}/{len(job_dicts)}".ljust(100), end="")
                    if job_dict["execution_time"] is None:
                        one_to_be_completed = True
                        job = service.job(job_dict["job_id"])
                        if job.in_final_state():
                            job_dict["execution_time"] = job.metrics()["timestamps"]["finished"]
                            if job.status() == "DONE":
                                job_dict["ev"] = float(job.result()[0].data.evs[0])
            self.save()
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
                backends_name_arr.append(backend.name)
            else:
                backends_name_arr.append(backend)
        return backends_name_arr
    
    def _date_range_to_datetime_range(self, date_range):
        return [datetime.fromisoformat(datestr) if type(datestr)==str else datestr for datestr in date_range]

    def _search_index_by_params(self, nqubits_arr=None, depths_arr=None, backends_arr=None, estimator_opt_dicts=None, test_circuit_name_func=False, observable_name_func=None, date_range=None, limit=None):        
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
        if backends_arr is not None:
            backends_name_arr = self._backends_objs_to_names(backends_arr)
            search_functions.append(lambda test: test["backends"] == sorted(backends_name_arr))
        if estimator_opt_dicts is not None:
            def estimator_opt_dicts_searchf(test):
                test_estimator_opt_dicts = []
                for job_dicts_arr in test["jobs"].values():
                    for job_dict in job_dicts_arr:
                        this_estimator_opt = job_dict["estimator_opt_dict"]
                        if this_estimator_opt not in test_estimator_opt_dicts:
                            test_estimator_opt_dicts.append(this_estimator_opt)
                return any([test_opt_dict in estimator_opt_dicts for test_opt_dict in test_estimator_opt_dicts])
            search_functions.append(estimator_opt_dicts_searchf)                
        if test_circuit_name_func is not None:
            test_circuit_name = test_circuit_name_func.__name__ if callable(test_circuit_name_func) else test_circuit_name_func
            search_functions.append(lambda test: test["test_circuit_name"] == test_circuit_name)
        if observable_name_func is not None:
            observable_name = observable_name_func.__name__ if callable(observable_name_func) else observable_name_func
            search_functions.append(lambda test: test["observable_func_name"] == observable_name)
        if date_range is not None:
            date_range = self._date_range_to_datetime_range(date_range)
            def date_range_searchf(test):
                execution_dates = []
                for job_dicts_arr in test["jobs"].values():
                    for job_dict in job_dicts_arr:
                        if (exectime := job_dict["execution_time"]) is not None:
                            execution_dates.append(datetime.fromisoformat(exectime))
                if len(execution_dates) > 0:
                    max_exec_time = max(execution_dates)
                    min_exec_time = min(execution_dates)
                    return (date_range[0] <= min_exec_time) and (date_range[1] >= max_exec_time)
                else:
                    return False
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
    
    def search_by_params(self, nqubits_arr=None, depths_arr=None, backends_arr=None, estimator_opt_dicts=None, test_circuit_name_func=None, observable_name_func=None, date_range=None, limit=None):
        indices = self._search_index_by_params(nqubits_arr, depths_arr, backends_arr, estimator_opt_dicts, test_circuit_name_func, observable_name_func, date_range, limit)
        return [self._data[i] for i in indices]
    
    def search_by_id(self, id):
        ind = self._search_batch_index_by_id(id)
        return self._data[ind]

    def plot_mean_error_by_date(self, nqubits_arr, depths_arr, backends_arr, estimator_opt_dicts, logical_circuit_generating_func, observable_generating_func, test_circuit_name=None, observable_name=None, date_range=None, simulator_max_bond_dimension=256, simulation_results_folder_path=None, plot_filepath=None, print_mode=False):
        # Create and optimize the circuits to run (For performance reasons, the result will be the same for the simulator)
        nqubits_arr = set(nqubits_arr)
        depths_arr = set(depths_arr)
        barrier_pm = PassManager([RemoveBarriers()])
        pm = generate_preset_pass_manager(optimization_level=2)
        circuits = []
        simulator_optimized_circuits = []
        for nqubits in nqubits_arr:
            for depth in depths_arr:
                if print_mode: print(f"\rGenerating circuits / nqubits: {nqubits} / depth: {depth}".ljust(100), end="")
                circuits.append(this_circuit := logical_circuit_generating_func(nqubits, depth))
                this_opt_circuit = pm.run(barrier_pm.run(this_circuit))
                simulator_optimized_circuits.append(this_opt_circuit)

        # Simulate the circuits
        perfect_evs = []
        for estimator_opt_dict in estimator_opt_dicts:
            estimator_options = {"default_precision": 1/np.sqrt(estimator_opt_dict["default_shots"])}
            simulator_options = {
                "method": "matrix_product_state",
                "matrix_product_state_max_bond_dimension": simulator_max_bond_dimension,
                "matrix_product_state_truncation_threshold": 1e-10
            }
            physical_circuit_name = logical_circuit_generating_func.__name__ if test_circuit_name is None else test_circuit_name
            observable_name = observable_generating_func.__name__ if observable_name is None else observable_name
            results_filename = f"benchmark_simresults_nqubits_{min(nqubits_arr)}-{max(nqubits_arr)}_depths_{min(depths_arr)}-{max(depths_arr)}_testcirc_{physical_circuit_name}_obsname_{observable_name}_bd_{simulator_max_bond_dimension}_shots_{estimator_opt_dict["default_shots"]}.txt"
            results_filepath = os.path.join("" if simulation_results_folder_path is None else simulation_results_folder_path, results_filename)
            if not os.path.isfile(results_filepath):
                if print_mode: print(f"\rSimulating circuits for shots = {estimator_opt_dict["default_shots"]}".ljust(100), end="")
                simulated_circuit_depths = [circuit.depth() for circuit in simulator_optimized_circuits]
                to_return_evs = np.ones(len(simulated_circuit_depths))
                to_run = [circuit for circuit in simulator_optimized_circuits if circuit.depth() > 0]
                simulated_jobs = sexec.execute_simulation_estimator_batch(simulator_options, estimator_options, to_run, observable_generating_func)
                simulated_evs = np.array([job.result()[0].data.evs[0] for job in simulated_jobs])
                to_return_evs[np.greater(simulated_circuit_depths, 0)] = simulated_evs
                perfect_evs += list(to_return_evs)
                if simulation_results_folder_path is not None:
                    np.savetxt(results_filepath, simulated_evs)
        
        # Get observables for each number of qubits and depths
        backends_name_arr = self._backends_objs_to_names(backends_arr)
        print(f"\rSearching dabatase".ljust(100), end="")
        found_benchmarks = self.search_by_params(nqubits_arr, depths_arr, backends_arr, estimator_opt_dicts, test_circuit_name, observable_name, date_range)
        estimator_opt_dicts_str = [json.dumps(estimator_opt_dict) for estimator_opt_dict in estimator_opt_dicts]
        measurement_dates = {x:{} for x in product(nqubits_arr, depths_arr, estimator_opt_dicts_str)}
        measured_evs = {x:{} for x in product(nqubits_arr, depths_arr, estimator_opt_dicts_str)}
        for nqubits in nqubits_arr:
            for depth in depths_arr:
                if print_mode: print(f"\rRetrieving jobs for nqubits = {nqubits} / depth = {depth}".ljust(100), end="")
                for i, estimator_opt_dict_str in enumerate(estimator_opt_dicts_str):
                    this_measurement_dates = {backend_name: [] for backend_name in backends_name_arr}
                    this_measured_evs = {backend_name: [] for backend_name in backends_name_arr}
                    for benchmark in found_benchmarks:
                        for backend_name, jobs_dicts in benchmark["jobs"].items():
                            for job_dict in jobs_dicts:
                                if (job_dict["nqubits"] == nqubits) and (job_dict["depth"] == depth) and (job_dict["estimator_opt_dict"] == estimator_opt_dicts[i]):
                                    if (job_dict["execution_time"] is not None) and (job_dict["ev"] is not None):
                                        this_measurement_dates[backend_name].append(datetime.fromisoformat(job_dict["execution_time"]))
                                        this_measured_evs[backend_name].append(job_dict["ev"])
                    measurement_dates[(nqubits, depth, estimator_opt_dict_str)] = this_measurement_dates
                    measured_evs[(nqubits, depth, estimator_opt_dict_str)] = this_measured_evs

        if print_mode: print("\r".ljust(100), end="")

        # Plot the results
        fig, axs, = plt.subplots(nrows=len(nqubits_arr)*len(depths_arr)*len(estimator_opt_dicts), figsize=[10, 12*len(nqubits_arr)*len(depths_arr)], sharex=True)
        njobs = len(nqubits_arr)*len(depths_arr)
        for k, estimator_opt_dict_str in enumerate(estimator_opt_dicts_str):
            for i, nqubits in enumerate(nqubits_arr):
                for j, depth in enumerate(depths_arr):
                    for backend_name in backends_name_arr:
                        errors = np.abs(np.array(measured_evs[(nqubits, depth, estimator_opt_dict_str)][backend_name]) - perfect_evs[i+j+k*njobs])
                        date_locator = pltdates.AutoDateLocator()
                        date_formatter = pltdates.ConciseDateFormatter(date_locator, formats=['%Y', '%d/%b', '%d/%b', '%H:%M', '%H:%M', '%S.%f'], offset_formats=['', '%Y', '%b/%Y', '%d-%b-%Y', '%d/%b/%Y', '%d/%b/%Y %H:%M'])
                        axs[i+j+k*njobs].plot(measurement_dates[(nqubits, depth, estimator_opt_dict_str)][backend_name], errors, "o-", linewidth=2, markersize=8, label=backend_name)
                        axs[i+j+k*njobs].xaxis.set_major_locator(date_locator)
                        axs[i+j+k*njobs].xaxis.set_major_formatter(date_formatter)
                        axs[i+j+k*njobs].tick_params(axis="x", rotation=45,) #ha="right")
                        axs[i+j+k*njobs].set_ylabel("Error")
                        axs[i+j+k*njobs].set_title(f"Qubits: {nqubits} / Depth: {depth} / Options: {k + 1}")
                        axs[i+j+k*njobs].legend()
                        axs[i+j+k*njobs].grid()
        plt.tight_layout()
        if plot_filepath is not None:
            plt.savefig(plot_filepath, dpi=300, facecolor="none")

def check_and_convert_to_unique_list(arg):
    try:
        arg = sorted(list(set(arg)))
    except TypeError:
        arg = [arg]
    return arg

def is_benchmark_running_mac(password=None):
    if password is None: password = getpass.getpass(prompt="Introduce superuser password")
    sp = subprocess.run(["sudo", "-S", "launchctl", "list"], input=f"{password}\n", text=True, capture_output=True)
    return "com.local.quantum.benchmark" in sp.stdout

def stop_benchmark_mac(password=None):
    if password is None: password = getpass.getpass(prompt="Introduce superuser password")
    subprocess.run(["sudo", "-S", "launchctl", "bootout", "system/com.local.quantum.benchmark"], input=f"{password}\n", text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def generate_and_execute_launchctl_file_mac(python_script, *args, seconds_start_interval=None, start_relativedeltas=None, stdout_path=None, stderr_path=None, working_directory=None, folder="/Library/LaunchDaemons", password=None):
    if password is None: password = getpass.getpass(prompt="Introduce superuser password")
    if is_benchmark_running_mac(password):
        stop_benchmark_mac(password)
    
    data = dict(
    Label="com.local.quantum.benchmark",
    ProgramArguments=[sys.executable, os.path.abspath(python_script)] + list(args) + ["-p", password],
    StandardOutPath=f"{os.path.abspath(stdout_path if stdout_path is not None else "benchmarks.log")}",
    StandardErrorPath=f"{os.path.abspath(stderr_path if stdout_path is not None else "benchmarks.err")}",
    WorkingDirectory=f"{os.path.abspath(working_directory) if working_directory is not None else os.getcwd()}")
    # KeepAlive={"PathState":{os.path.abspath(python_script):True}})

    if seconds_start_interval is not None: data["StartInterval"] = seconds_start_interval

    if start_relativedeltas is not None:
        try:
            start_relativedeltas = list(start_relativedeltas)
        except TypeError:
            start_relativedeltas = [start_relativedeltas]
        start_calendar_dicts = []
        for start_relativedelta in start_relativedeltas:
            this_start_calendar = {}
            if (min := start_relativedelta.minute) is not None: this_start_calendar["Minute"] = int(min)
            if (hour := start_relativedelta.hour) is not None: this_start_calendar["Hour"] = int(hour)
            if (day := start_relativedelta.day) is not None:
                if 1 <= day <= 31: 
                    this_start_calendar["Day"] = int(day)
                else:
                    raise ValueError("Days in start_relativedelta must be in range [1, 31]")
            if (weekday := start_relativedelta.weekday) is not None: this_start_calendar["weekday"] = int(weekday.weekday + 1)
            if (month := start_relativedelta.month) is not None:
                if 1 <= month <= 12:
                    this_start_calendar["Month"] = int(month)
                else:
                    raise ValueError("Months in start_relativedelta must be in range [1, 12]")
            start_calendar_dicts.append(this_start_calendar)
        data["StartCalendarInterval"] = start_calendar_dicts
    try:
        with open(finalpath := os.path.join(folder, "com.local.quantum.benchmark.plist"), "wb") as f:
            plistlib.dump(data, f)
    except PermissionError as e:
        with open(ogpath := os.path.abspath("com.local.quantum.benchmark.plist"), "wb") as f:
            plistlib.dump(data, f)
        subprocess.run(["sudo", "-S", "mv", ogpath, finalpath], input=f"{password}\n", text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["sudo", "-S", "chown", "root:wheel", finalpath], input=f"{password}\n", text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["sudo", "-S", "launchctl", "load", finalpath], input=f"{password}\n", text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def check_benchmark_status_mac(stdout_path=None, password=None):
    if password is None: password = getpass.getpass(prompt="Introduce superuser password")
    if is_benchmark_running_mac(password):
        with open("/Library/LaunchDaemons/com.local.quantum.benchmark.plist", "rb") as f:
            data = plistlib.load(f)
        settings_str = json.dumps(data, indent=4)
        print("The benchmark IS running")
        print("SETTINGS:")
        print(settings_str)
        if stdout_path is not None:
            print("LOGS:")
            with open(stdout_path, "r") as f:
                lines = f.readlines()
                last_ten_lines = lines[len(lines)-11::-1]
                print("\n".join(last_ten_lines))
    else:
        print("The benchmark is NOT running")