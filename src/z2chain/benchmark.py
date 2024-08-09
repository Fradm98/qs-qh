from z2chain.circs import particle_pair_quench_simulation_circuits
from z2chain.operators import pauli_zs_mean, pauli_xs_mean
from qiskit_ibm_runtime import QiskitRuntimeService
from utils.benchmarking import BenchmarkDB
from utils.circs import compute_uncompute
from functools import partial
import datetime
import argparse

def get_z2_chain_benchmark_circuit(chain_length, J, h, lamb, t_compute, layers, x_basis=False):
    compute_circuit = particle_pair_quench_simulation_circuits(chain_length, J, h, lamb, chain_length//2 - 1, 1, t_compute, layers, layers, x_basis, False)[1]
    benchmark_circuit = compute_uncompute(compute_circuit, barrier=True)
    return benchmark_circuit

def get_mean_occupation(nqubits, x_basis=False):
    if x_basis:
        return pauli_xs_mean(nqubits)
    else:
        return pauli_zs_mean(nqubits)
    
def update_and_send_jobs(benchmarkdb, chain_length_arr, layers_arr, devices_arr, service, J, h, lamb, t_compute, x_basis=False, shots=4096):
    benchmarkdb.update_status(service)
    z2_chain_benchmark_circuit = lambda chain_length, layers: get_z2_chain_benchmark_circuit(chain_length, J, h, lamb, t_compute, layers, x_basis)
    benchmarkdb.execute(chain_length_arr, layers_arr, devices_arr, service, z2_chain_benchmark_circuit, pauli_xs_mean if x_basis else pauli_zs_mean, shots=shots, test_circuit_name="z2_chain_trotter_compuncomp", observable_name="mean_paulis_" + ("x" if x_basis else "z"))

def main():
    parser = argparse.ArgumentParser(prog="Z2 chain IBMQ Benchmark", description="Sends a benchmark circuit and updates database")
    parser.add_argument("benchmark_database", help="Path of the benchmark database file", type=str)
    parser.add_argument("ibmkey", help="IBM Quantum API token", type=str)
    parser.add_argument("-i", "--instance", help="IBM Quantum instance identifier", type=str, default=None)
    parser.add_argument("-n", "--length", help="Lengths of the chain for the benchmark circuits", nargs="+", type=int, required=True)
    parser.add_argument("-d", "--layers", help="Number of layers of the benchmark circuits", nargs="+", type=int, required=True)
    parser.add_argument("-b", "--backends", help="Backends to execute the benchmark", nargs="+", type=str, required=True)
    parser.add_argument("-t", "--tcompute", help="Time of the compute part of the benchmark circuit", type=float, required=True)
    parser.add_argument("-j", "--jparam", help="Value of the J parameter in the Z2 Hamiltonian", type=float, required=True)
    parser.add_argument("-g", "--hparam", help="Value of the h parameter in the Z2 hamiltonian", type=float, required=True)
    parser.add_argument("-l", "--lambparam", help="Value of the λ parameter in the Z2 hamiltonian", type=float, required=True)
    parser.add_argument("-x", "--xbasis", help="Work in x-basis", action="store_true")
    parser.add_argument("-s", "--shots", help="Number of shots for the beanchmark circuits", default=4096)
    parser.add_argument("-q", "--quiet", help="Do not log output", action="store_true")

    args = parser.parse_args()
    bdb = BenchmarkDB(args.benchmark_database)
    service = QiskitRuntimeService(channel="ibm_quantum", token=args.ibmkey, instance=args.instance)
    try:
        update_and_send_jobs(bdb, args.length, args.layers, args.backends, service, args.jparam, args.hparam, args.lambparam, args.tcompute, args.xbasis, args.shots)
        if not args.quiet: print(f"{datetime.datetime.now()}: Sent job and updated database")
    except Exception as e: 
        if not args.quiet: print(f"{datetime.datetime.now()}: Error ocurred")
        raise e

if __name__ == "__main__":
    main()