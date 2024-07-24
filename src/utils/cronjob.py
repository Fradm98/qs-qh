import json
from functools import partial
from .benchmarking import benchmarkdb
from z2chain.circs import physical_particle_pair_quench_simulation_circuits

args = {
    "nqubits_arr": 8,
    "depths_arr": 15,
    "backends_arr": "ibm_fez",
    "physical_circuit_generating_func": [],
    "observable_generating_funcs": [],
    "shots": 4096,
    "test_circuit_name": None,
    "observable_name": None
}


# Read arguments from JSON file
with open('args_benchmarking.json', 'r') as f:
    args = json.load(f)

# Extract arguments
arg1 = args.get("nqubits_arr")
arg2 = args.get("depths_arr")
arg3 = args.get("backends_arr")
arg4 = args.get("physical_circuit_generating_func")
arg5 = args.get("observable_generating_funcs")
arg6 = args.get("shots")
arg7 = args.get("test_circuit_name")
arg8 = args.get("observable_name")

def logical_circuit_func(chain_length, depth):
    return physical_particle_pair_quench_simulation_circuits(chain_length=chain_length, layers=depth, J=1, h=0.05, lamb=0.5, particle_pair_left_position=chain_length//2-1, particle_pair_length=1, final_time=5, layout=None)
    
# Execute the benchmark
benchmarkdb.execute(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
