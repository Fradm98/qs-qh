from qiskit_ibm_runtime import QiskitRuntimeService
import json

def call_ibm_account():
    try:
        service = QiskitRuntimeService(name="ibm-ikerbasque")
    except:
        print("Use the token from IBM quantum ikerbasque collab and save the account")
        service = QiskitRuntimeService(channel="ibm_quantum", token="0fb1cd48a37e2722ff37089427a3d8616d7b2287a908d94dfb62520a17e05294adc3458b0ddcd2333774eb5497b16f75965bbb0233cf9c3b0df7343ffa1374b9")  # Credentials may be needed
        service.save_account(channel="ibm_quantum", token="0fb1cd48a37e2722ff37089427a3d8616d7b2287a908d94dfb62520a17e05294adc3458b0ddcd2333774eb5497b16f75965bbb0233cf9c3b0df7343ffa1374b9", name="ibm-ikerbasque", overwrite=True)
        print(f"{[key for key, value in service.saved_accounts().items()]} account saved")
    return service

def least_busy_backend():
    service = call_ibm_account()
    print(service.backends(simulator=False))
    lb = service.least_busy(simulator=False)
    print(lb.name)
    backend = service.get_backend(name=lb.name)
    return backend

def save_jobs(filename: str, jobs: dict):
    with open(f'{filename}.json', 'w') as json_file:
        json.dump(jobs, json_file, indent=4)

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
