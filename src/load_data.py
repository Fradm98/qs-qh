from z2hex.circs import simulation_and_odr_circuits, get_layouts
from z2hex.geometry import HeavyHexLattice, sort_layouts_by_func, measurement_decoherence_layout_cost
from utils.postselection_multithreaded import load_postselected_jobs_obs
from z2hex.operators import local_pauli_z, postselection_operators
from z2hex.geometry import HeavyHexLattice
from z2hex.gdd import zero_mean
from utils.hexec import ExecDB
import numpy as np
import pickle
import os

def load_single_string_data():
    datas = []
    tn_datas = []
    
    # Col1
    print("Loading col1 data")
    device = "ibm_kingston"
    lattice_dims = (2, 2)
    J = 5
    h = 2
    lamb = 1
    string = [4, 10, 17, 24, 30]
    final_time = 4.5
    dt = 0.15
    steps = 36

    x_basis = False

    fractional_gates = False

    confidence = 0.7

    odr = True
    postselect_odr = True
    flip_threshold = 1
    all_or_first_layout_qubit = None
    forbidden_qubits = []
    max_layouts = 5
    recovery = True
    g_func = zero_mean

    pt_num_randomizations = 40
    pt_shots_per_randomization = 2500

    shots = 300000
    max_shots = 100000
    min_shots = 10000

    sampler_options = {
        "default_shots": shots,
        "execution": {
            "init_qubits": True,
            "meas_type": "classified"
        },
        "dynamical_decoupling": {
            "enable": False,
            "sequence_type": "XY4"
        },
        "twirling": {
            "enable_gates": True,
            "enable_measure": True,
            "num_randomizations": pt_num_randomizations,
            "shots_per_randomization": pt_shots_per_randomization
        }
    }

    execution_database = "../data/job_dbs/1string.json"
    jobs_result_folder = os.path.splitext(execution_database)[0]
    job_db = ExecDB(execution_database)
    curated_data_folder_path = "../data/curated_data/"

    with open("../data/circuits/1string/col1.pkl", "rb") as f:
        circuits, odr_circuits = pickle.load(f)

    lattice = HeavyHexLattice(*lattice_dims)
    extra_options = {"backend": device, "plaquette_width": lattice_dims[0], "plaquette_height": lattice_dims[1], "J": np.array(J).tolist(), "h": np.array(h).tolist(), "λ":np.array(lamb).tolist(), "g": g_func.__name__ if callable(g_func) else g_func, "string_arr": np.array(string).tolist(), "final_time": final_time, "dt": dt, "steps": steps, "first_qubit": all_or_first_layout_qubit, "x_basis": x_basis, "max_shots": max_shots} | ({"fractional_gates": True} if fractional_gates else {})
    observable_generating_funcs = [(lambda nqubits, i=i: local_pauli_z(lattice, i)) for i in range(len(lattice))]
    postselection_operators_func = lambda nqubits: postselection_operators(lattice, x_basis=False)

    job_index = 0

    overwrite = False
    if curated_data_folder_path is not None:
        curated_data_filepath = os.path.join(curated_data_folder_path, f"z2hexquench_curated_data_{device}_px_{lattice_dims[0]}_py_{lattice_dims[1]}_maxt_{final_time}_dt_{dt}_steps_{steps}_J_{J:.04f}_h_{h:.04f}_lamb_{lamb:.04f}_g_{g_func.__name__ if callable(g_func) else g_func}_shots_{shots}_fqb_{all_or_first_layout_qubit}_string_{"-".join([str(i) for i in string[:10]])}_xbasis_{x_basis}_dd_{sampler_options["dynamical_decoupling"]["enable"]}_pt_{sampler_options["twirling"]["enable_gates"]}_ft_{flip_threshold}_pODR_{postselect_odr}_confidence_{confidence}.pkl")
    else:
        curated_data_filepath = ""

    if overwrite or not os.path.exists(curated_data_filepath):
        if odr:
            observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts = load_postselected_jobs_obs(job_db, None, sampler_options, circuits, postselection_operators_func, observable_generating_funcs, extra_options=extra_options, odr_circuits=odr_circuits, postselect_odr=postselect_odr, recovery=recovery, flip_threshold=flip_threshold, jobs_result_folder=jobs_result_folder, job_index=job_index, confidence=confidence, return_samples_dicts=True, return_postselected_samples_dicts=True, return_layouts=True, verbose=True, max_shots=max_shots, minimum_shots=min_shots)
        else:
            observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts = load_postselected_jobs_obs(job_db, None, sampler_options, circuits, postselection_operators_func, observable_generating_funcs, extra_options=extra_options, recovery=recovery, flip_threshold=flip_threshold, jobs_result_folder=jobs_result_folder, job_index=job_index, confidence=confidence, return_samples_dicts=True, return_postselected_samples_dicts=True, return_layouts=True, verbose=True, max_shots=max_shots, minimmum_shots=min_shots)
        if curated_data_filepath is not None: 
            with open(curated_data_filepath, "wb") as f:
                pickle.dump((observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts), f)
    else:
        with open(curated_data_filepath, "rb") as f:
            observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts = pickle.load(f)

    datas.append([observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts])
    tn_datas.append(np.loadtxt("../data/tn/1string/col1_tndata.txt"))

    # Col 2 (Load curated data directly)
    print("Loading col2 data")
    curated_data_filepath = "../data/curated_data/col2_hdata_large_old_confidence_0.7.pkl"
    with open(curated_data_filepath, "rb") as f:
        data = pickle.load(f)
        datas.append(data)
    tn_datas.append(np.loadtxt("../data/tn/1string/col2_tndata_large.txt"))

    # Col 3
    print("Loading col3 data")
    device = "ibm_kingston"
    lattice_dims = (3, 3)
    J = 0.3
    h = 0.5
    lamb = 1
    string = [4, 14, 24, 34, 45, 53, 63]
    final_time = 5
    dt = 0.15
    steps = 34

    x_basis = False

    fractional_gates = False

    confidence = 0.7

    odr = True
    postselect_odr = True
    flip_threshold = 1
    all_or_first_layout_qubit = None
    forbidden_qubits = []
    max_layouts = 5
    recovery = True
    g_func = zero_mean

    pt_num_randomizations = 50
    pt_shots_per_randomization = 2000

    shots = 480000
    max_shots = 100000
    min_shots = 10000

    sampler_options = {
        "default_shots": shots,
        "execution": {
            "init_qubits": True,
            "meas_type": "classified"
        },
        "dynamical_decoupling": {
            "enable": False,
            "sequence_type": "XY4"
        },
        "twirling": {
            "enable_gates": True,
            "enable_measure": True,
            "num_randomizations": pt_num_randomizations,
            "shots_per_randomization": pt_shots_per_randomization
        }
    }

    execution_database = "../data/job_dbs/1string.json"
    jobs_result_folder = os.path.splitext(execution_database)[0]
    job_db = ExecDB(execution_database)
    curated_data_folder_path = "../data/curated_data/"

    with open("../data/circuits/1string/col3.pkl", "rb") as f:
        circuits, odr_circuits = pickle.load(f)

    lattice = HeavyHexLattice(*lattice_dims)
    extra_options = {"backend": device, "plaquette_width": lattice_dims[0], "plaquette_height": lattice_dims[1], "J": np.array(J).tolist(), "h": np.array(h).tolist(), "λ":np.array(lamb).tolist(), "g": g_func.__name__ if callable(g_func) else g_func, "string_arr": np.array(string).tolist(), "final_time": final_time, "dt": dt, "steps": steps, "first_qubit": all_or_first_layout_qubit, "x_basis": x_basis, "max_shots": max_shots} | ({"fractional_gates": True} if fractional_gates else {})
    observable_generating_funcs = [(lambda nqubits, i=i: local_pauli_z(lattice, i)) for i in range(len(lattice))]
    postselection_operators_func = lambda nqubits: postselection_operators(lattice, x_basis=False)

    job_index = 0

    overwrite = False
    if curated_data_folder_path is not None:
        curated_data_filepath = os.path.join(curated_data_folder_path, f"z2hexquench_curated_data_{device}_px_{lattice_dims[0]}_py_{lattice_dims[1]}_maxt_{final_time}_dt_{dt}_steps_{steps}_J_{J:.04f}_h_{h:.04f}_lamb_{lamb:.04f}_g_{g_func.__name__ if callable(g_func) else g_func}_shots_{shots}_fqb_{all_or_first_layout_qubit}_string_{"-".join([str(i) for i in string[:10]])}_xbasis_{x_basis}_dd_{sampler_options["dynamical_decoupling"]["enable"]}_pt_{sampler_options["twirling"]["enable_gates"]}_ft_{flip_threshold}_pODR_{postselect_odr}_confidence_{confidence}.pkl")
    else:
        curated_data_filepath = ""

    if overwrite or not os.path.exists(curated_data_filepath):
        if odr:
            observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts = load_postselected_jobs_obs(job_db, None, sampler_options, circuits, postselection_operators_func, observable_generating_funcs, extra_options=extra_options, odr_circuits=odr_circuits, postselect_odr=postselect_odr, recovery=recovery, flip_threshold=flip_threshold, jobs_result_folder=jobs_result_folder, job_index=job_index, confidence=confidence, return_samples_dicts=True, return_postselected_samples_dicts=True, return_layouts=True, verbose=True, max_shots=max_shots, minimum_shots=min_shots)
        else:
            observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts = load_postselected_jobs_obs(job_db, None, sampler_options, circuits, postselection_operators_func, observable_generating_funcs, extra_options=extra_options, recovery=recovery, flip_threshold=flip_threshold, jobs_result_folder=jobs_result_folder, job_index=job_index, confidence=confidence, return_samples_dicts=True, return_postselected_samples_dicts=True, return_layouts=True, verbose=True, max_shots=max_shots, minimmum_shots=min_shots)
        if curated_data_filepath is not None: 
            with open(curated_data_filepath, "wb") as f:
                pickle.dump((observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts), f)
    else:
        with open(curated_data_filepath, "rb") as f:
            observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts = pickle.load(f)

    datas.append([observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts])
    tn_datas.append(np.loadtxt("../data/tn/1string/col3_tndata_m_0.3_g_0.5.txt"))

    return datas, tn_datas

def load_benchmark_data():
    em_hdatas = []

    # Raw
    print("Loading benchmark raw data")
    device = "ibm_basquecountry"
    lattice_dims = (3, 7)
    J = 0
    h = 0
    lamb = 1
    string = [27, 26, 24, 22, 33, 41, 43, 45, 53, 64, 62, 60, 71, 79, 81, 83, 91, 102, 100, 98, 109, 117, 119, 121, 122] # Benchmark
    final_time = 7.5
    dt = 0.25
    steps = 60

    x_basis = False

    fractional_gates = False

    confidence = 0.7

    odr = True
    postselect_odr = True
    flip_threshold = 1
    all_or_first_layout_qubit = None
    forbidden_qubits = []
    max_layouts = 5
    recovery = True
    g_func = None

    pt_num_randomizations = 120
    pt_shots_per_randomization = 2500

    shots = 480000
    max_shots = 100000
    min_shots = 10000

    sampler_options = {
        "default_shots": shots,
        "execution": {
            "init_qubits": True,
            "meas_type": "classified"
        },
        "dynamical_decoupling": {
            "enable": False,
            "sequence_type": "XY4"
        },
        "twirling": {
            "enable_gates": False,
            "enable_measure": False,
            "num_randomizations": pt_num_randomizations,
            "shots_per_randomization": pt_shots_per_randomization
        }
    }

    execution_database = "../data/job_dbs/benchmark.json"
    jobs_result_folder = os.path.splitext(execution_database)[0]
    job_db = ExecDB(execution_database)
    curated_data_folder_path = "../data/curated_data/"

    with open("../data/circuits/benchmark/raw.pkl", "rb") as f:
        circuits, odr_circuits = pickle.load(f)

    lattice = HeavyHexLattice(*lattice_dims)
    extra_options = {"backend": device, "plaquette_width": lattice_dims[0], "plaquette_height": lattice_dims[1], "J": np.array(J).tolist(), "h": np.array(h).tolist(), "λ":np.array(lamb).tolist(), "g": g_func.__name__ if callable(g_func) else g_func, "string_arr": np.array(string).tolist(), "final_time": final_time, "dt": dt, "steps": steps, "first_qubit": all_or_first_layout_qubit, "x_basis": x_basis, "max_shots": max_shots} | ({"fractional_gates": True} if fractional_gates else {})
    observable_generating_funcs = [(lambda nqubits, i=i: local_pauli_z(lattice, i)) for i in range(len(lattice))]
    postselection_operators_func = lambda nqubits: postselection_operators(lattice, x_basis=False)

    job_index = 0

    overwrite = False
    if curated_data_folder_path is not None:
        curated_data_filepath = os.path.join(curated_data_folder_path, f"z2hexquench_curated_data_{device}_px_{lattice_dims[0]}_py_{lattice_dims[1]}_maxt_{final_time}_dt_{dt}_steps_{steps}_J_{J:.04f}_h_{h:.04f}_lamb_{lamb:.04f}_g_{g_func.__name__ if callable(g_func) else g_func}_shots_{shots}_fqb_{all_or_first_layout_qubit}_string_{"-".join([str(i) for i in string[:10]])}_xbasis_{x_basis}_dd_{sampler_options["dynamical_decoupling"]["enable"]}_pt_{sampler_options["twirling"]["enable_gates"]}_ft_{flip_threshold}_pODR_{postselect_odr}_confidence_{confidence}.pkl")
    else:
        curated_data_filepath = ""

    if overwrite or not os.path.exists(curated_data_filepath):
        if odr:
            observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts = load_postselected_jobs_obs(job_db, None, sampler_options, circuits, postselection_operators_func, observable_generating_funcs, extra_options=extra_options, odr_circuits=odr_circuits, postselect_odr=postselect_odr, recovery=recovery, flip_threshold=flip_threshold, jobs_result_folder=jobs_result_folder, job_index=job_index, confidence=confidence, return_samples_dicts=True, return_postselected_samples_dicts=True, return_layouts=True, verbose=True, max_shots=max_shots, minimum_shots=min_shots)
        else:
            observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts = load_postselected_jobs_obs(job_db, None, sampler_options, circuits, postselection_operators_func, observable_generating_funcs, extra_options=extra_options, recovery=recovery, flip_threshold=flip_threshold, jobs_result_folder=jobs_result_folder, job_index=job_index, confidence=confidence, return_samples_dicts=True, return_postselected_samples_dicts=True, return_layouts=True, verbose=True, max_shots=max_shots, minimmum_shots=min_shots)
        if curated_data_filepath is not None: 
            with open(curated_data_filepath, "wb") as f:
                pickle.dump((observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts), f)
    else:
        with open(curated_data_filepath, "rb") as f:
            observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts = pickle.load(f)

    em_hdatas.append([observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts])

    # GDD
    print("Loading benchmark GDD data")
    device = "ibm_basquecountry"
    lattice_dims = (3, 7)
    J = 0
    h = 0
    lamb = 1
    string = [27, 26, 24, 22, 33, 41, 43, 45, 53, 64, 62, 60, 71, 79, 81, 83, 91, 102, 100, 98, 109, 117, 119, 121, 122] # Benchmark
    final_time = 7.5
    dt = 0.25
    steps = 60

    x_basis = False

    fractional_gates = False

    confidence = 0.7

    odr = True
    postselect_odr = True
    flip_threshold = 1
    all_or_first_layout_qubit = None
    forbidden_qubits = []
    max_layouts = 5
    recovery = True
    g_func = zero_mean

    pt_num_randomizations = 120
    pt_shots_per_randomization = 2500

    shots = 480000
    max_shots = 100000
    min_shots = 10000

    sampler_options = {
        "default_shots": shots,
        "execution": {
            "init_qubits": True,
            "meas_type": "classified"
        },
        "dynamical_decoupling": {
            "enable": False,
            "sequence_type": "XY4"
        },
        "twirling": {
            "enable_gates": False,
            "enable_measure": False,
            "num_randomizations": pt_num_randomizations,
            "shots_per_randomization": pt_shots_per_randomization
        }
    }

    execution_database = "../data/job_dbs/benchmark.json"
    jobs_result_folder = os.path.splitext(execution_database)[0]
    job_db = ExecDB(execution_database)
    curated_data_folder_path = "../data/curated_data/"

    with open("../data/circuits/benchmark/raw.pkl", "rb") as f:
        circuits, odr_circuits = pickle.load(f)

    lattice = HeavyHexLattice(*lattice_dims)
    extra_options = {"backend": device, "plaquette_width": lattice_dims[0], "plaquette_height": lattice_dims[1], "J": np.array(J).tolist(), "h": np.array(h).tolist(), "λ":np.array(lamb).tolist(), "g": g_func.__name__ if callable(g_func) else g_func, "string_arr": np.array(string).tolist(), "final_time": final_time, "dt": dt, "steps": steps, "first_qubit": all_or_first_layout_qubit, "x_basis": x_basis, "max_shots": max_shots} | ({"fractional_gates": True} if fractional_gates else {})
    observable_generating_funcs = [(lambda nqubits, i=i: local_pauli_z(lattice, i)) for i in range(len(lattice))]
    postselection_operators_func = lambda nqubits: postselection_operators(lattice, x_basis=False)

    job_index = 0

    overwrite = False
    if curated_data_folder_path is not None:
        curated_data_filepath = os.path.join(curated_data_folder_path, f"z2hexquench_curated_data_{device}_px_{lattice_dims[0]}_py_{lattice_dims[1]}_maxt_{final_time}_dt_{dt}_steps_{steps}_J_{J:.04f}_h_{h:.04f}_lamb_{lamb:.04f}_g_{g_func.__name__ if callable(g_func) else g_func}_shots_{shots}_fqb_{all_or_first_layout_qubit}_string_{"-".join([str(i) for i in string[:10]])}_xbasis_{x_basis}_dd_{sampler_options["dynamical_decoupling"]["enable"]}_pt_{sampler_options["twirling"]["enable_gates"]}_ft_{flip_threshold}_pODR_{postselect_odr}_confidence_{confidence}.pkl")
    else:
        curated_data_filepath = ""

    if overwrite or not os.path.exists(curated_data_filepath):
        if odr:
            observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts = load_postselected_jobs_obs(job_db, None, sampler_options, circuits, postselection_operators_func, observable_generating_funcs, extra_options=extra_options, odr_circuits=odr_circuits, postselect_odr=postselect_odr, recovery=recovery, flip_threshold=flip_threshold, jobs_result_folder=jobs_result_folder, job_index=job_index, confidence=confidence, return_samples_dicts=True, return_postselected_samples_dicts=True, return_layouts=True, verbose=True, max_shots=max_shots, minimum_shots=min_shots)
        else:
            observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts = load_postselected_jobs_obs(job_db, None, sampler_options, circuits, postselection_operators_func, observable_generating_funcs, extra_options=extra_options, recovery=recovery, flip_threshold=flip_threshold, jobs_result_folder=jobs_result_folder, job_index=job_index, confidence=confidence, return_samples_dicts=True, return_postselected_samples_dicts=True, return_layouts=True, verbose=True, max_shots=max_shots, minimmum_shots=min_shots)
        if curated_data_filepath is not None: 
            with open(curated_data_filepath, "wb") as f:
                pickle.dump((observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts), f)
    else:
        with open(curated_data_filepath, "rb") as f:
            observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts = pickle.load(f)

    em_hdatas.append([observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts])

    # PT
    print("Loading benchmark PT data")
    device = "ibm_basquecountry"
    lattice_dims = (3, 7)
    J = 0
    h = 0
    lamb = 1
    string = [27, 26, 24, 22, 33, 41, 43, 45, 53, 64, 62, 60, 71, 79, 81, 83, 91, 102, 100, 98, 109, 117, 119, 121, 122] # Benchmark
    final_time = 7.5
    dt = 0.25
    steps = 60

    x_basis = False

    fractional_gates = False

    confidence = 0.7

    odr = True
    postselect_odr = True
    flip_threshold = 1
    all_or_first_layout_qubit = None
    forbidden_qubits = []
    max_layouts = 5
    recovery = True
    g_func = False

    pt_num_randomizations = 120
    pt_shots_per_randomization = 2500

    shots = 480000
    max_shots = 100000
    min_shots = 10000

    sampler_options = {
        "default_shots": shots,
        "execution": {
            "init_qubits": True,
            "meas_type": "classified"
        },
        "dynamical_decoupling": {
            "enable": False,
            "sequence_type": "XY4"
        },
        "twirling": {
            "enable_gates": True,
            "enable_measure": True,
            "num_randomizations": pt_num_randomizations,
            "shots_per_randomization": pt_shots_per_randomization
        }
    }

    execution_database = "../data/job_dbs/benchmark.json"
    jobs_result_folder = os.path.splitext(execution_database)[0]
    job_db = ExecDB(execution_database)
    curated_data_folder_path = "../data/curated_data/"

    with open("../data/circuits/benchmark/raw.pkl", "rb") as f:
        circuits, odr_circuits = pickle.load(f)

    lattice = HeavyHexLattice(*lattice_dims)
    extra_options = {"backend": device, "plaquette_width": lattice_dims[0], "plaquette_height": lattice_dims[1], "J": np.array(J).tolist(), "h": np.array(h).tolist(), "λ":np.array(lamb).tolist(), "g": g_func.__name__ if callable(g_func) else g_func, "string_arr": np.array(string).tolist(), "final_time": final_time, "dt": dt, "steps": steps, "first_qubit": all_or_first_layout_qubit, "x_basis": x_basis, "max_shots": max_shots} | ({"fractional_gates": True} if fractional_gates else {})
    observable_generating_funcs = [(lambda nqubits, i=i: local_pauli_z(lattice, i)) for i in range(len(lattice))]
    postselection_operators_func = lambda nqubits: postselection_operators(lattice, x_basis=False)

    job_index = 0

    overwrite = False
    if curated_data_folder_path is not None:
        curated_data_filepath = os.path.join(curated_data_folder_path, f"z2hexquench_curated_data_{device}_px_{lattice_dims[0]}_py_{lattice_dims[1]}_maxt_{final_time}_dt_{dt}_steps_{steps}_J_{J:.04f}_h_{h:.04f}_lamb_{lamb:.04f}_g_{g_func.__name__ if callable(g_func) else g_func}_shots_{shots}_fqb_{all_or_first_layout_qubit}_string_{"-".join([str(i) for i in string[:10]])}_xbasis_{x_basis}_dd_{sampler_options["dynamical_decoupling"]["enable"]}_pt_{sampler_options["twirling"]["enable_gates"]}_ft_{flip_threshold}_pODR_{postselect_odr}_confidence_{confidence}.pkl")
    else:
        curated_data_filepath = ""

    if overwrite or not os.path.exists(curated_data_filepath):
        if odr:
            observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts = load_postselected_jobs_obs(job_db, None, sampler_options, circuits, postselection_operators_func, observable_generating_funcs, extra_options=extra_options, odr_circuits=odr_circuits, postselect_odr=postselect_odr, recovery=recovery, flip_threshold=flip_threshold, jobs_result_folder=jobs_result_folder, job_index=job_index, confidence=confidence, return_samples_dicts=True, return_postselected_samples_dicts=True, return_layouts=True, verbose=True, max_shots=max_shots, minimum_shots=min_shots)
        else:
            observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts = load_postselected_jobs_obs(job_db, None, sampler_options, circuits, postselection_operators_func, observable_generating_funcs, extra_options=extra_options, recovery=recovery, flip_threshold=flip_threshold, jobs_result_folder=jobs_result_folder, job_index=job_index, confidence=confidence, return_samples_dicts=True, return_postselected_samples_dicts=True, return_layouts=True, verbose=True, max_shots=max_shots, minimmum_shots=min_shots)
        if curated_data_filepath is not None: 
            with open(curated_data_filepath, "wb") as f:
                pickle.dump((observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts), f)
    else:
        with open(curated_data_filepath, "rb") as f:
            observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts = pickle.load(f)

    em_hdatas.append([observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts])

    # GDD PT
    print("Loading benchmark GDD PT data")
    device = "ibm_basquecountry"
    lattice_dims = (3, 7)
    J = 0
    h = 0
    lamb = 1
    string = [27, 26, 24, 22, 33, 41, 43, 45, 53, 64, 62, 60, 71, 79, 81, 83, 91, 102, 100, 98, 109, 117, 119, 121, 122] # Benchmark
    final_time = 7.5
    dt = 0.25
    steps = 60

    x_basis = False

    fractional_gates = False

    confidence = 0.7

    odr = True
    postselect_odr = True
    flip_threshold = 1
    all_or_first_layout_qubit = None
    forbidden_qubits = []
    max_layouts = 5
    recovery = True
    g_func = zero_mean

    pt_num_randomizations = 120
    pt_shots_per_randomization = 2500

    shots = 480000
    max_shots = 100000
    min_shots = 10000

    sampler_options = {
        "default_shots": shots,
        "execution": {
            "init_qubits": True,
            "meas_type": "classified"
        },
        "dynamical_decoupling": {
            "enable": False,
            "sequence_type": "XY4"
        },
        "twirling": {
            "enable_gates": True,
            "enable_measure": True,
            "num_randomizations": pt_num_randomizations,
            "shots_per_randomization": pt_shots_per_randomization
        }
    }

    execution_database = "../data/job_dbs/benchmark.json"
    jobs_result_folder = os.path.splitext(execution_database)[0]
    job_db = ExecDB(execution_database)
    curated_data_folder_path = "../data/curated_data/"

    with open("../data/circuits/benchmark/raw.pkl", "rb") as f:
        circuits, odr_circuits = pickle.load(f)

    lattice = HeavyHexLattice(*lattice_dims)
    extra_options = {"backend": device, "plaquette_width": lattice_dims[0], "plaquette_height": lattice_dims[1], "J": np.array(J).tolist(), "h": np.array(h).tolist(), "λ":np.array(lamb).tolist(), "g": g_func.__name__ if callable(g_func) else g_func, "string_arr": np.array(string).tolist(), "final_time": final_time, "dt": dt, "steps": steps, "first_qubit": all_or_first_layout_qubit, "x_basis": x_basis, "max_shots": max_shots} | ({"fractional_gates": True} if fractional_gates else {})
    observable_generating_funcs = [(lambda nqubits, i=i: local_pauli_z(lattice, i)) for i in range(len(lattice))]
    postselection_operators_func = lambda nqubits: postselection_operators(lattice, x_basis=False)

    job_index = 0

    overwrite = False
    if curated_data_folder_path is not None:
        curated_data_filepath = os.path.join(curated_data_folder_path, f"z2hexquench_curated_data_{device}_px_{lattice_dims[0]}_py_{lattice_dims[1]}_maxt_{final_time}_dt_{dt}_steps_{steps}_J_{J:.04f}_h_{h:.04f}_lamb_{lamb:.04f}_g_{g_func.__name__ if callable(g_func) else g_func}_shots_{shots}_fqb_{all_or_first_layout_qubit}_string_{"-".join([str(i) for i in string[:10]])}_xbasis_{x_basis}_dd_{sampler_options["dynamical_decoupling"]["enable"]}_pt_{sampler_options["twirling"]["enable_gates"]}_ft_{flip_threshold}_pODR_{postselect_odr}_confidence_{confidence}.pkl")
    else:
        curated_data_filepath = ""

    if overwrite or not os.path.exists(curated_data_filepath):
        if odr:
            observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts = load_postselected_jobs_obs(job_db, None, sampler_options, circuits, postselection_operators_func, observable_generating_funcs, extra_options=extra_options, odr_circuits=odr_circuits, postselect_odr=postselect_odr, recovery=recovery, flip_threshold=flip_threshold, jobs_result_folder=jobs_result_folder, job_index=job_index, confidence=confidence, return_samples_dicts=True, return_postselected_samples_dicts=True, return_layouts=True, verbose=True, max_shots=max_shots, minimum_shots=min_shots)
        else:
            observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts = load_postselected_jobs_obs(job_db, None, sampler_options, circuits, postselection_operators_func, observable_generating_funcs, extra_options=extra_options, recovery=recovery, flip_threshold=flip_threshold, jobs_result_folder=jobs_result_folder, job_index=job_index, confidence=confidence, return_samples_dicts=True, return_postselected_samples_dicts=True, return_layouts=True, verbose=True, max_shots=max_shots, minimum_shots=min_shots)
        if curated_data_filepath is not None: 
            with open(curated_data_filepath, "wb") as f:
                pickle.dump((observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts), f)
    else:
        with open(curated_data_filepath, "rb") as f:
            observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts = pickle.load(f)

    em_hdatas.append([observables, observables_confidence, samples_dicts, postselected_samples_dicts, layouts])

    # Trotter data
    filepaths_trotter = ["../data/curated_data/Trotter benchmark data/dt_0.1.pkl", "../data/curated_data/Trotter benchmark data/dt_0.125.pkl", "../data/curated_data/Trotter benchmark data/dt_0.15.pkl", "../data/curated_data/Trotter benchmark data/dt_0.175.pkl", "../data/curated_data/Trotter benchmark data/dt_0.2.pkl"]
    hardware_datas_trotter = []
    for filepath in filepaths_trotter:
        with open(filepath, "rb") as f:
            hardware_datas_trotter.append(pickle.load(f))

    # 1D data
    filepaths_hardware_contour = ["../data/curated_data/1D_benchmark_data/m_5_g_0.8.pkl", "../data/curated_data/1D_benchmark_data/m_5_g_0.01.pkl"]
    hardware_datas_contour = []
    for filepath in filepaths_hardware_contour:
        with open(filepath, "rb") as f:
            hardware_datas_contour.append(pickle.load(f))

    filepaths_hardware_1d = ["../data/curated_data/1D_benchmark_data/2nd_run_close/m_5_g_0.8_nonan.pkl", "../data/curated_data/1D_benchmark_data/2nd_run_close/m_5_g_0.8_clifford_nonan.pkl", "../data/curated_data/1D_benchmark_data/2nd_run_close/m_5_g_0.01.pkl", "../data/curated_data/1D_benchmark_data/2nd_run_close/m_5_g_0.01_clifford.pkl"]
    hardware_datas_1d = []
    for filepath in filepaths_hardware_1d:
        with open(filepath, "rb") as f:
            hardware_datas_1d.append(pickle.load(f))

    # TN data
    filename_tn_nm = "../data/tn/benchmark/trotter/data.txt"
    filenames_tn_1d = ["../data/tn/benchmark/1d/tn_m_5_g_0.8.npy", "../data/tn/benchmark/1d/tn_m_5_g_0.01.npy"]
    
    tn_observables_nm = np.loadtxt(filename_tn_nm)
    tn_observables_1d = []
    for filename in filenames_tn_1d:
        tn_observables_1d.append(np.load(filename))

    return [em_hdatas, hardware_datas_trotter, hardware_datas_contour, hardware_datas_1d], [tn_observables_nm, tn_observables_1d]