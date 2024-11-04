from qiskit.primitives.base.base_primitive_job import BasePrimitiveJob
from qiskit.primitives.primitive_job import PrimitiveJob
import numpy as np

def convert_jobs_to_site_gauge_matrix(jobs_arr):
    printed_warning_header = False
    first_success_index = 0
    while first_success_index < len(jobs_arr):
        try:
            first_success_job = jobs_arr[first_success_index]
            if not first_success_job.in_final_state() and type(first_success_job) is not PrimitiveJob:
                raise TimeoutError("Some jobs are still to be finished")
            else:
                first_success_result = first_success_job.result()[0].data.evs[::-1]
            site_gauge_observable_matrix = np.full((len(jobs_arr), len(first_success_result)), np.nan)
            break
        except:
            if not printed_warning_header: 
                print("WARNING: Some jobs failed, the plot will be incomplete\nFAILED JOBS\nIndex | ID")
                printed_warning_header = True
            print(f"{str(first_success_index).ljust(6)}| {jobs_arr[first_success_index].job_id()}")
            first_success_index += 1
            continue
    site_gauge_observable_matrix[first_success_index] = (1 - first_success_result[::-1])/2
    for i, job in enumerate(jobs_arr[first_success_index::], start=first_success_index):
        try:
            if not job.in_final_state() and type(first_success_job) is not PrimitiveJob:
                raise TimeoutError("Some jobs are still to be finished")
            else:
                this_job_result = job.result()[0].data.evs[::-1]
            site_gauge_observable_matrix[i] = (1 - this_job_result)/2
        except:
            if not printed_warning_header: 
                print("WARNING: Some jobs failed, the plot will be incomplete\nFAILED JOBS\nIndex | ID")
                printed_warning_header = True
            print(f"{str(i).ljust(6)}| {job.job_id()}")
            continue
    return site_gauge_observable_matrix[:i]

def save_site_gauge_observable_matrix(site_gauge_observable_matrix, filepath, header=""):
    if isinstance(site_gauge_observable_matrix[0], BasePrimitiveJob):
        site_gauge_observable_matrix = convert_jobs_to_site_gauge_matrix(site_gauge_observable_matrix)

    np.savetxt(filepath, site_gauge_observable_matrix, header=header)

def load_site_gauge_observable_matrix(filepath):
    return np.loadtxt(filepath)