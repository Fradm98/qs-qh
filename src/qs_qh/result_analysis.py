import json
import numpy as np

def retrieve_job_id(n_qubits, depth, extrapolation_method, mem, filename):
    ids = []
    with open(f'{filename}.json', 'r') as json_file:
        jobs = json.load(json_file)
        for job in jobs:
            if job['n_qubits'] == n_qubits and job['depth'] == depth:
                for c_dict in job['batch']:
                    if c_dict['extrapolator'] == extrapolation_method and c_dict['mem'] == mem:
                        ids.append(c_dict['job_id'])
    return ids

def retrieve_results(qubits, depths, zne_extrapolator, mem, zne, service):
    mean_errors = []
    for qubit in qubits:
        for depth in depths:
            job_ids = retrieve_job_id(qubit, depth, zne_extrapolator, mem, f"logs/jobs_qubits_{qubits}_depths_{depths}_pauli-Z_weight_1_stat_shots_3_zne_{zne}_mem_{mem}")
            res = []
            for job_id in job_ids:
                job = service.job(job_id=job_id)
                print(job)
                result = job.result()[0]
                ev = result.data.evs.tolist()
                res.append(ev)
                print(f"- {ev} ({zne_extrapolator})")
            mean = np.mean(np.asarray(res))
            std = np.std(np.asarray(res))
            mean_errors.append(mean)
    return mean_errors