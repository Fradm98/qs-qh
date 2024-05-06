from qiskit_ibm_runtime import QiskitRuntimeService
import json

def call_ibm_account():
    try:
        service = QiskitRuntimeService()
    except:
        print("Use the token from IBM Industry session and save the account")
        service = QiskitRuntimeService(channel="ibm_quantum", token="4463e0fe7fb2e7d70ba9dfe7eebe7ffc7ef8bf2c57cd2c74f40af32e8b063534f46e1e794ee505523a7d5ca10b20a742b0f0b2306a8959f200d0154c91a36a6c")  # Credentials may be needed
        service.save_account(channel="ibm_quantum", token="4463e0fe7fb2e7d70ba9dfe7eebe7ffc7ef8bf2c57cd2c74f40af32e8b063534f46e1e794ee505523a7d5ca10b20a742b0f0b2306a8959f200d0154c91a36a6c", name="ibm_industry_session", overwrite=True)
        print(f"{[key for key, value in service.saved_accounts().items()]} account saved")
    return service

def least_busy_backend():
    service = call_ibm_account()
    print(service.backends(simulator=False))
    lb = service.least_busy(simulator=False)
    print(lb)
    backend = service.get_backend(name=lb.name)
    return backend

def save_jobs(filename: str, jobs: dict):
    with open(f'{filename}.json', 'w') as json_file:
        json.dump(jobs, json_file, indent=4)

def retrieve_job_id(n_qubits, depth, extrapolation_method):
    with open('jobs.json', 'r') as json_file:
        jobs = json.load(json_file)
        for job in jobs:
            print(job)
            if job['n_qubits'] == n_qubits and job['depth'] == depth:
                for c_dict in job['batch']:
                    if c_dict['extrapolator'] == extrapolation_method:
                        return c_dict['job_id']