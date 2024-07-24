from functools import partial
from utils.benchmarking import benchmarkdb
from z2chain.circs import particle_pair_quench_simulation_circuits
from z2chain.operators import local_pauli_z
from qiskit_ibm_runtime import QiskitRuntimeService
# from cobos.tokens import ibm_token, group_instance
from fradm.tokens import ibm_token, group_instance

def logical_circuit_func(chain_length, depth):
    return particle_pair_quench_simulation_circuits(chain_length=chain_length, 
                                                    layers=depth, 
                                                    J=1, h=0.05, lamb=0.5, 
                                                    particle_pair_left_position=chain_length//2-1, 
                                                    particle_pair_length=1, 
                                                    final_time=5, 
                                                    measure_every_layers=1)[1]

def observable_func(L):
    observable_name = "Occupation numbers"
    return [partial(local_pauli_z, qubit_ind=i) for i in range(2*L-1)]

channel = "ibm_quantum"
service = QiskitRuntimeService(channel=channel, token=ibm_token, instance=group_instance)

# args = {
#     "nqubits_arr": [8, 16, 32, 64],
#     "depths_arr": [3, 5, 7, 9, 11, 13, 15],
#     "backends_arr": ["ibm_fez", 
#                      "ibm_torino", 
#                      "ibm_kyiv", 
#                      "ibm_sherbrooke", 
#                      "ibm_brussels", 
#                      "ibm_kyoto", 
#                      "ibm_nazca", 
#                      "ibm_osaka", 
#                      "ibm_brisbane", 
#                      "ibm_strasbourg", 
#                      "ibm_cusco"],
#     "logical_circuit_generating_func": logical_circuit_func,
#     "observable_generating_funcs": observable_func,
#     "shots": 4096,
#     "test_circuit_name": None,
#     "observable_name": None
# }

args = {
    "nqubits_arr": [8],
    "depths_arr": [3],
    "devices_arr": ["ibm_fez", 
                     "ibm_kyiv", 
                     "ibm_strasbourg"],
    "service": service,
    "logical_circuit_generating_func": logical_circuit_func,
    "observable_generating_funcs": observable_func,
    "shots": 4096,
    "test_circuit_name": None,
    "observable_name": None,
    
}


# Extract arguments
arg1 = args.get("nqubits_arr")
arg2 = args.get("depths_arr")
arg3 = args.get("devices_arr")
arg4 = args.get("service")
arg5 = args.get("logical_circuit_generating_func")
arg6 = args.get("observable_generating_funcs")
arg7 = args.get("shots")
arg8 = args.get("test_circuit_name")
arg9 = args.get("observable_name")

# Execute the benchmark
path = "test_benchmark.json"
bench_db = benchmarkdb(path=path)
bench_db.execute(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
