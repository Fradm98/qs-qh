"""Script for qiskit serverless
   
   1. - Creates a session, submits one callibration job and then transpiles circuits
   3. - This is the function to be uploaded to qiskit functions
   2. - To be executed with a qiskit serverless job database
"""

# Imports mirrored from the principal repo via symbolic links to upload only the
# required files to Qiskit Serverless

from utils.postselection import measure_diagonal_observables, get_recovered_postselected_samples_dict, execute_postselected_sampler_batch, pauli_to_str
from circs import odr_compuncomp_circuits, odr_compuncomp_circuits_eplg, erradj_particle_pair_quench_simulation_circuits_eplg
from operators import local_pauli_x, local_pauli_z, postselection_operators
from utils.circs import check_and_measure_active_qubits

# Qiskit is installed in the Qiskit service 💁🏽

from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2
from utils.hexec import get_backend_best_qubit_chain, ExecDB
from qiskit_serverless import get_arguments, save_result
from functools import partial
import numpy as np

def execute_noise_cal_quench_session(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, steps, max_layers, backend, optimization_level, sampler_options, g=None, layout=None, odr=True, x_basis=False, max_shots=75000, flip_threshold=4, debug=False):
   # Open a session, execute the test circuit to get the error to adjust,
   # transpile the error adjusted time evolution circuits and submit
   # jobs

   print("Running...")

   # Get session details
   service = QiskitRuntimeService(channel="ibm_quantum")
   backend = service.backend(backend)

   # Define postselection operators and observables
   observable_generating_func = [partial(local_pauli_x if x_basis else local_pauli_z, qubit_ind=i) for i in range(2*chain_length-1)]
   postselection_operators_func = lambda nqubits: postselection_operators((nqubits + 1)//2, x_basis)
   observables = [obsfunc(2*chain_length-1) for obsfunc in observable_generating_func]
   postselection_operators = postselection_operators(chain_length, x_basis)

   print("Generated postselection operators and observables")
   
   # Transpile the test circuit
   layout = get_backend_best_qubit_chain(backend, 2*chain_length-1) if layout is None else layout
   cal_circ_func_args = (chain_length, J, h, lamb, final_time/max_layers, 1, backend, g, particle_pair_left_position, particle_pair_length, layout, x_basis)
   cal_circ = odr_compuncomp_circuits(*cal_circ_func_args)[-1]

   print("Transpiled calibration circuit")

   # Open the session and sumbit the test circuit
   with Session(backend=backend) as session:
      print("Session open")
      
      sampler = SamplerV2(mode=session, options=sampler_options)
      measured_cal_circ = check_and_measure_active_qubits(cal_circ)
      if not debug: job_cal = sampler.run(measured_cal_circ)

      print("Calibration circuit submitted, waiting for results..." + ("(Dummy, debug mode)" if debug else ""))

      # Wait for the job to finish and measure the error
      if not debug:
         cal_samples_dict = list(job_cal.result()[0].data.values())[0].get_counts()
         cal_postselected_samples_dict = get_recovered_postselected_samples_dict(cal_samples_dict, postselection_operators, layout, flip_threshold=flip_threshold)
         cal_observables = measure_diagonal_observables(cal_postselected_samples_dict, observables, layout)
         mean_observable = cal_observables.mean(axis=1)
         error = 1 - np.sqrt(mean_observable)
      else:
         error = 0.1

      print("Calibration result received, transpiling simulation circuits..." + ("(Dummy, debug mode)" if debug else ""))

      # Transpile the error adjusted circuits and submit the new jobs
      circ_func_args = (chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, steps, backend, optimization_level, error, g, layout, x_basis, True, False, max_layers)
      physical_circuits = erradj_particle_pair_quench_simulation_circuits_eplg(*circ_func_args)
      if odr:
         odr_func_args = (chain_length, J, h, lamb, final_time, steps, backend, error, g, particle_pair_left_position, particle_pair_length, layout, x_basis, False, max_layers)
         odr_circuits = odr_compuncomp_circuits_eplg(*odr_func_args)
         circuits = [odr_circuits[i//2] if i % 2 == 0 else physical_circuits[i//2] for i in range(2*len(physical_circuits))]
         if not debug:
            jobs = execute_postselected_sampler_batch(backend, sampler_options, physical_circuits, postselection_operators_func, observable_generating_func, odr_circuits, max_shots=max_shots)
         else:
            jobs = ["debug"]
      else:
         circuits = physical_circuits
         if not debug:
            jobs = execute_postselected_sampler_batch(backend, sampler_options, physical_circuits, postselection_operators_func, observable_generating_func, max_shots=max_shots)   
         else:
            jobs = ["debug"]

      print("Submitting simulation jobs..." + ("(Dummy, debug mode)" if debug else ""))

      extra_options = {"backend": backend, "L": chain_length, "J": J, "h": h, "λ":lamb, "g": g, "particle_pair_position": particle_pair_left_position, "particle_pair_length": particle_pair_length, "final_time": final_time, "noise": error}
      postselection_strings = sorted([pauli_to_str(op) for op in postselection_operators])
      observables_strings = sorted([pauli_to_str(op) for op in set(observables)])
      post_obs_info_dict = {"postselection_ops": postselection_strings, "observables": observables_strings}
      options = extra_options | post_obs_info_dict | sampler_options | {"odr": odr}
      if not debug:
         jobs_ids = [job.job_id() for job in jobs]
         dbdata = ExecDB.generate_batch_db_data(options, circuits, "Sampler", jobs_ids)
      else:
         dbdata = {"debug": True}

   debug_str = " (Dummy, debug mode)" if debug else ""
   print(f"Simulation jobs in queue{debug_str}. Function ended.")

   # Return the job ids and the calibration error
   return jobs, dbdata

def main():
   arguments = get_arguments()
   chain_length = arguments.get("L")
   J = arguments.get("J")
   h = arguments.get("h")
   lamb = arguments.get("lamb")
   pp_ps = arguments.get("particle_pair_left_position")
   pp_length = arguments.get("particle_pair_left_position")
   final_time = arguments.get("final_time")
   steps = arguments.get("steps")
   max_layers = arguments.get("max_layers")
   backend = arguments.get("backend")
   optimization_level = arguments.get("optimization_level")
   sampler_options = arguments.get("sampler_options")
   g = arguments.get("g", None)
   layout = arguments.get("layout", None)
   odr = arguments.get("odr", False)
   x_basis = arguments.get("x_basis", False)
   max_shots = arguments.get("max_shots", 75000)
   flip_threshold = arguments.get("flip_threshold", 4)
   debug = arguments.get("debug", False)

   jobs, dbdata = execute_noise_cal_quench_session(chain_length, J, h, lamb, pp_ps, pp_length, final_time, steps, max_layers, backend, optimization_level, sampler_options, g, layout, odr, x_basis, max_shots, flip_threshold, debug)
   save_result({"jobs": jobs, "dbdata": dbdata})

main()