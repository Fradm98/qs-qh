# Guide for postselected hardware jobs

The notebook `z2_chain_hardware_postselection` in this same folder is used to submit and measure the results of postselected quantum simulations. To perform postselected quantum simulations, we run the circuits using the sampler and discard all the states that we find not to be +1 eigenstates of all the Pauli strings in a given set. We call this set the _postselection operators_. We require every postselection operator to be diagonal in certain Pauli basis Z, X, Y. This is to avoid the introduction of extremely long circuits in order to measure all the postselection conditions simultaneously. All the code required for this task is contained in `src/utils/postselection.py`.

## Using the notebook

### Simulation options

All the options for the 1D quench simulation are set in the cell _Simulation settings_.

- `L_arr`: Chain sizes. If this variable is a list, a batch of jobs will be submitted for a given size.

- `J`, `h`, `lamb`: Constants of the Chain's Hamiltonian

$$
H = \underbrace{-\frac{J}{\lambda} \sum_{n} \tau_{n}^z}_{\text{Mass term}} - \underbrace{\frac{h}{\lambda} \sum_{(n, v)} \sigma_{n, v}^{z}}_{\text{Electric energy}} - \underbrace{\sum_{n, v} \tau_{n + v}^x \sigma_{(n, v)}^x \tau_{n}^x}_{\text{Matter-gauge interaction}}
$$

- `g`: Strength of Gauge dynamical decoupling. If `None`, the gauge terms are not included in the evolution.

$$
\tilde{H} = H + \sum_n G_n.
$$

- `x_basis`: Whether to work in the $X$ basis.

- `particle_pair_left_position`: Index of the left node of the particle pair in the initial state.

- `final_time`: Final time for the temporal evolution.

- `steps`: Number of points in the interval $[0, \mathrm{final\_time}]$ for which to measure observables.

- `shots`: Number of shots *per set of commuting observables*. If this variable is a list, a batch of jobs will be submitted for a given size.

- `execution_database`: Filepath of the jobs database (.json format)

- `jobs_result_folder`: Path of the folder to save the downloaded jobs

- `circ_folder_path`: Path to folder to save the circuits. **WARNING: Layout is saved in the circuits, generate new ones for new jobs submissions**

### Device selection and circuit transpilation

These two cells must be ran always, either to load or submit jobs.

- The eplg_absolute in `Device selection` is only used when submitting jobs. The job database saves the value of this variable when jobs were submitted for job loading.

- When loading jobs is useful to use saved circuits in order to avoid long transpilation times. This is done using the `generate_and_save_circs` function in the `Circuit transpilation, observables & postselection` cell instead of `erradj_particle_pair_quench_simulation_circuits`. Comment one of the lines setting the `physical_circuits` variable in order to switch behavior. The reason for this is that circuits are passed as arguments when searching for jobs because they are basically containers of some of the job information. The layout of the circuits is saved independently in the database. So it does not matter if it is the same when loading the circuits.

### Send jobs to hardware

After running the options, device and circuit cells. The _send jobs to hardware_ cell can be run to submit jobs to hardware seamlessly. A batch of jobs is submitted per chain size and number of shots. Each batch contains one job per `step` times $(\#\text{ of non-diagonal operators + 1})$. We call diagonal operators to those that can be expressed exclusively as Pauli strings that are diagonal in the same basis as the postselection operators. The reasoning for this is that, although all the postselection operators and observables that make sense to measure in our models can be measured simultaneously, very long circuits may be required for that. However, we are interested at most in 3-local observables and the circuits to measure each of them individually are doable (depth $\sim3$). We then separate the non-diagonal observables to be measured independently.

### Load jobs

Jobs are found in the database by the options set in the options cell. Just run the _load jobs with the same options_ to load them in the jobs variable. This is passed to the x_t_plot function directly to generate the plots. The first time the jobs are downloaded, they are saved as files into the `jobs_result_folder` if specified. They are named as follows: `<job_id>.json`.

### Plots

The plotting functions in the last two cells are implemented in `src/z2chain/plotting.py`. Just pass the arguments with descriptive names and they should work.