# README Circuits

In this document, we explain how we program our circuits in Qiskit.

We are simulating a second order Trotter evolution for the dynamics of $\mathbb{Z}_2$-Higgs model. Each of the Trotter steps of the trotterized evolution consists on the following operators 

$$U_T(t) = e^{-i H_M t/2N} e^{-i H_I t/N} e^{-i H_M t/2N}$$

With:

$$H_M = -\frac{J}{\lambda} \sum_{n} \tau_{n}^z - \frac{h}{\lambda} \sum_{(n, v)} \sigma_{n, v}^{z}$$

$$H_I = -\sum_{n, v} \tau_{n + v}^x \sigma_{(n, v)}^x \tau_{n}^x$$

The exponential of $H_M$ consists on single qubit rotations, while the exponential of $H_I$ can be inplemented with 2*`connectivity` two qubit gates, where `connectivity` is the number of gauge links connected to each matter site in the $\mathbb{Z}_2$-Higgs lattice.

The complete time evolution of the system is implemented by repeating the action of these operators as many time as desired.

The Qiskit circuits implementing this trotterized time evolution can be found in `src/z2chain/circs.py`. We now outline our approach to programtically generate these circuits.

## Implementing each of the exponentials for a single Trotter step

We programatically generate the circuits of each of the exponentials on a single Trotter steps and then plug them together in order. The inputs for the functions generating these circuit is the length of the chain, the parameters of the Hamiltonian and the desired time of evolution **for each Trotter step** $t_\ell$. In all functions, we add an `x_basis` argument to work in the basis where the vacuum state for $\lambda=0$ is $\ket{+}$, with the subsequent rotation applied to the Hamiltonian.

### $H_M$ exponential

The exponential of $H_M$ simply consists on single qubits rotations. The code is simply

```python
class TotalSingleBodyPropagator(QuantumCircuit):
    def __init__(self, chain_length, x_basis=False):
        nqubits = 2*chain_length - 1
        super().__init__(nqubits)
        t_tau = Parameter("t_τ")
        t_sigma = Parameter("t_σ")
        if x_basis:
            self.rx(-2*t_tau, np.arange(0, self.num_qubits, 2))
            self.rx(-2*t_sigma, np.arange(1, self.num_qubits, 2))
        else:
            self.rz(-2*t_tau, np.arange(0, self.num_qubits, 2))
            self.rz(-2*t_sigma, np.arange(1, self.num_qubits, 2))
```

The resulting circuit for `chain_length = 2` is the following

<p align="center"> 
    <img src="https://raw.githubusercontent.com/Fradm98/qs-qh/refs/heads/main/Module%20READMEs/Figures/total_single_body_propagator.svg">

### $H_I$ Hamiltonian

The exponential of $H_I$ is generated with the following code.

```python
class TotalInteractionPropagator(QuantumCircuit):
    def __init__(self, chain_length, x_basis=False):
        nqubits = 2*chain_length - 1
        first_cnots_trgt_qubits = np.arange(0, nqubits-1, 2)
        first_cnots_ctrl_qubits = np.arange(1, nqubits, 2)
        second_cnots_trgt_qubits = np.arange(2, nqubits, 2)
        t_lamb = Parameter("t_lamb")
        t_g = Parameter("t_g")
        super().__init__(nqubits)
        if x_basis:
            self.cx(first_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.cx(second_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.rz(-2*t_lamb, np.arange(1, nqubits, 2))
            self.cx(second_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.cx(first_cnots_trgt_qubits, first_cnots_ctrl_qubits)
        else:
            self.cx(first_cnots_ctrl_qubits, first_cnots_trgt_qubits)
            self.cx(first_cnots_ctrl_qubits, second_cnots_trgt_qubits)
            self.rx(-2*t_lamb, np.arange(1, nqubits, 2))
            self.cx(first_cnots_ctrl_qubits, second_cnots_trgt_qubits)
            self.cx(first_cnots_ctrl_qubits, first_cnots_trgt_qubits)
```

And the resulting circuit for `chain_length = 2` is the following

<p align="center"> 
    <img src="https://raw.githubusercontent.com/Fradm98/qs-qh/refs/heads/main/Module%20READMEs/Figures/total_interaction_propagator.svg">

To add the Gauge Dynamical decouplig term, we opt to create another class instead of passing an argument because otherwise, creating the parameters is a mess. These two classes are very similar.

```python
class TotalInteractionPropagatorDD(QuantumCircuit):
    def __init__(self, chain_length, x_basis=False):
        nqubits = 2*chain_length - 1
        first_cnots_trgt_qubits = np.arange(0, nqubits-1, 2)
        first_cnots_ctrl_qubits = np.arange(1, nqubits, 2)
        second_cnots_trgt_qubits = np.arange(2, nqubits, 2)
        t_lamb = Parameter("t_lamb")
        t_g = Parameter("t_g")
        super().__init__(nqubits)
        if x_basis:
            self.cx(first_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.cx(second_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.rx(-2*t_g, np.arange(0, nqubits, 2))
            self.rz(-2*t_lamb, np.arange(1, nqubits, 2))
            self.cx(second_cnots_trgt_qubits, first_cnots_ctrl_qubits)
            self.cx(first_cnots_trgt_qubits, first_cnots_ctrl_qubits)
        else:
            self.cx(first_cnots_ctrl_qubits, first_cnots_trgt_qubits)
            self.cx(first_cnots_ctrl_qubits, second_cnots_trgt_qubits)
            self.rz(-2*t_g, np.arange(0, nqubits, 2))
            self.rx(-2*t_lamb, np.arange(1, nqubits, 2))
            self.cx(first_cnots_ctrl_qubits, second_cnots_trgt_qubits)
            self.cx(first_cnots_ctrl_qubits, first_cnots_trgt_qubits)
```

The resulting circuit for `chain_length = 2` is the following

<p align="center"> 
    <img src="https://raw.githubusercontent.com/Fradm98/qs-qh/refs/heads/main/Module%20READMEs/Figures/total_interaction_propagator_gauge.svg">

## Plugging all the circuits together

To generate the physical circuits, we transpile one Trotter layer and then repeat the physical circuits. The reason for this is to pass a simpler circuit to the transpiler to avoid _very long_ transpiler times. We then add a custom simple 1-qubit simplification transpiler pass to automatically combine rotations at the beggining and end of each of the Trotter layers. For our simulations, we also generate complete logical circuits.

### Logical Trotter circuits

The code for generating the complete logical Trotter circuit is the following

```python
def SecondOrderTrotter(chain_length, J, h, lamb, t_total, layers, g=None, x_basis=False, barriers=False):
    t_layer = t_total/layers
    if g is None:
        total_interaction_propagator = TotalInteractionPropagator(chain_length, x_basis).decompose()
        total_interaction_propagator.assign_parameters([lamb*t_layer], inplace=True)
    else:
        total_interaction_propagator = TotalInteractionPropagatorDD(chain_length, x_basis)
        total_interaction_propagator.assign_parameters([g*t_layer, lamb*t_layer], inplace=True)
    total_single_body_propagator = TotalSingleBodyPropagator(chain_length, x_basis)
    total_single_body_propagator.assign_parameters([h*t_layer/2, t_layer*J/2], inplace=True)
    layer = total_single_body_propagator.compose(total_interaction_propagator).compose(total_single_body_propagator)
    if barriers: layer.barrier()
    return layer.repeat(layers).decompose()
```

And the resulting circuit for `chain_length = 2` is the following

<p align="center"> 
    <img src="https://raw.githubusercontent.com/Fradm98/qs-qh/refs/heads/main/Module%20READMEs/Figures/logical_trotter.svg">

### Physical circuits

As introduced above, to generate the physical (transpiled) circuits, we opt for transpiling a single Trotter layer and then repeating the transpiled circuit to generate the complete time evolution cirtuit. Additionally, we adjust the number of trotter layers by taking into account the circuit noise error present in the device (see [notes](https://github.com/Fradm98/qs-qh/blob/main/Notes/notes.pdf)).

The code for generating the complete physical Trotter circuit is the following (it includes state preparation)

```python
def erradj_particle_pair_quench_simulation_circuits(chain_length, J, h, lamb, particle_pair_left_position, particle_pair_length, final_time, steps, backend, optimization_level, eplg_absolute, g=None, layout=None, x_basis=False, barriers=False):
    layout = layout[:2*chain_length-1] if layout is not None else None
    t_arr = np.linspace(0, final_time, steps)
    nlayers_arr = get_erradj_number_of_trotter_layers(t_arr, eplg_absolute, trotter_order=2)
    t_perlayer_arr = t_arr / nlayers_arr
    state_preparation_logical_circ = particle_pair_initial_state(chain_length, particle_pair_left_position, particle_pair_length, x_basis=x_basis)
    trotter_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=layout)
    circs_to_return = []
    for i in range(steps-1):
        this_trotter_layer_logical_circuit = SecondOrderTrotter(chain_length, J, h, lamb, t_perlayer_arr[i+1], 1, g=g, x_basis=x_basis, barriers=barriers)
        this_trotter_layer_physical_circuit = trotter_pm.run(this_trotter_layer_logical_circuit)
        final_index_layout = this_trotter_layer_physical_circuit.layout.final_index_layout()
        state_prep_pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, initial_layout=final_index_layout)
        state_preparation_physical_circuit = state_prep_pm.run(state_preparation_logical_circ)
        this_time_physical_circuit = state_preparation_physical_circuit.compose(this_trotter_layer_physical_circuit.repeat(nlayers_arr[i+1])).decompose()
        this_time_physical_circuit = remove_idle_qwires(this_time_physical_circuit, relabeling=np.arange(len(final_index_layout))[np.argsort(final_index_layout)])
        layout_dict = {final_index_layout[i]:this_time_physical_circuit.qubits[i] for i in range(this_time_physical_circuit.num_qubits)}
        layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
        sqcancel_pm = PassManager([Optimize1qGates(target=backend.target), Optimize1qGatesDecomposition(target=backend.target), Optimize1qGatesSimpleCommutation(target=backend.target)])
        sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        this_time_physical_circuit = sqopt_pm.run(this_time_physical_circuit)
        if i == 0:
            measured_state_preparation_circuit = QuantumCircuit(2*chain_length-1, 2*chain_length-1)
            measured_state_preparation_circuit = measured_state_preparation_circuit.compose(state_preparation_logical_circ)
            state_preparation_physical_circuit = state_prep_pm.run(measured_state_preparation_circuit)
            state_preparation_physical_circuit.measure(np.array(final_index_layout)[np.argsort(final_index_layout)], np.argsort(np.array(final_index_layout)[np.argsort(final_index_layout)]))
            circs_to_return.append(state_preparation_physical_circuit)
        circs_to_return.append(this_time_physical_circuit)
    return circs_to_return
```

And the resulting circuit for `chain_length = 5` is the following

<p align="center"> 
    <img src="https://raw.githubusercontent.com/Fradm98/qs-qh/refs/heads/main/Module%20READMEs/Figures/physical_trotter.svg">

