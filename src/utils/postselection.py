from utils.circs import simplify_logical_circuits, count_non_idle_qubits, append_basis_change_circuit, join_transpiled_circuits
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import RuntimeDecoder, RuntimeEncoder
from qiskit.quantum_info import Pauli, PauliList
from utils.hexec import execute_sampler_batch
from qiskit import QuantumCircuit
from collections import Counter
from pymatching import Matching
import sympy as sym
import numpy as np
import json
import os

# -----------------------------------------
#           BASIS CHANGE CIRCUITS
# -----------------------------------------

# AUXILIARY FUNCTIONS

def pauli_to_str(pauli):
    if type(pauli) == Pauli:
        return "".join([str(opel) for opel in pauli])
    elif type(pauli) == str:
        return pauli
    else:
        raise ValueError("pauli argument is not a Pauli object")

def check_postselection_observable_commutation(observables, postselection_ops):
    postselection_ops = PauliList([Pauli(postselection_op) for postselection_op in postselection_ops])
    observables = PauliList([Pauli(observable) for observable in observables])
    nqubits = len(postselection_ops[0])
    for postselection_op in postselection_ops:
        this_nqubits = len(postselection_op)
        if this_nqubits != nqubits:
            raise ValueError("All operators must act on the same number of qubits")
    for observable in observables:
        this_nqubits = len(observable)
        if this_nqubits != nqubits:
            raise ValueError("All operators must act on the same number of qubits")
    return len(postselection_ops.commutes_with_all(observables)) == len(postselection_ops)

def count_non_diagonal_qubits(commuting_pauli_set):
    max_non_diagonal_qubits = 0
    for pauli in commuting_pauli_set:
        ndiagonal = pauli_to_str(pauli).count("I") + pauli_to_str(pauli).count("Z")
        this_non_diagonal_qubits = len(pauli) - ndiagonal
        if this_non_diagonal_qubits > max_non_diagonal_qubits:
            max_non_diagonal_qubits = this_non_diagonal_qubits
    return max_non_diagonal_qubits

def operators_affecting_qb_range(first, last, operators, return_operator_range=False):
    operators = PauliList(operators)
    affecting_operators = []
    first_affected_qb = last
    last_affected_qb = first
    for op in operators:
        op_str = pauli_to_str(op)
        sub_op_string = op_str[first:last+1]
        present_paulis = set(sub_op_string)
        if len(present_paulis - {"I"}) > 0:
            affecting_operators.append(op)
            if return_operator_range:
                non_identity_indices = [i for i, c in enumerate(op_str) if c != "I"]
                this_first_affected_qb = min(non_identity_indices)
                this_last_affected_qb = max(non_identity_indices)
                if this_first_affected_qb < first_affected_qb:
                    first_affected_qb = this_first_affected_qb
                if this_last_affected_qb > last_affected_qb:
                    last_affected_qb = this_last_affected_qb
    if return_operator_range:
        return PauliList(affecting_operators), (first_affected_qb, last_affected_qb)
    else:
        return PauliList(affecting_operators)

def diagonalization_susceptible_qb_range(observables, postselection_ops, return_operators=False):
    if type(observables) != list:
        observables = [observables]
    operator_ranges = np.zeros((len(observables), 2), dtype=int)
    if return_operators: operators_affecting_obs_range = {}
    for i, observable in enumerate(observables):
        observable_str = pauli_to_str(Pauli(observable))
        first_obs_qb = len(observable_str)
        last_obs_qb = 0
        for j, c in enumerate(observable_str):
            if c in ["X", "Y"]:
                if j < first_obs_qb:
                    first_obs_qb = j
                elif j > last_obs_qb:
                    last_obs_qb = j
        operators_affecting_obs_range, this_operator_range = operators_affecting_qb_range(first_obs_qb, last_obs_qb, observable + postselection_ops, return_operator_range=True)
        operator_ranges[i, :] = this_operator_range
        if return_operators: operators_affecting_obs_range += set(operators_affecting_obs_range)
    operator_ranges = (np.min(operator_ranges[:, 0]), np.max(operator_ranges[:, 1]))
    if return_operators:
        return operator_ranges, PauliList(operators_affecting_obs_range)
    else:
        return operator_ranges
    
def is_diagonal(pauli, basis="Z"):
    if not ((type(pauli) != Pauli) or (type(pauli) != str)):
        raise ValueError("This function only work for Pauli objects or strings")
    pauli_str = pauli_to_str(pauli)
    for c in pauli_str:
        if c not in ["I", basis]:
            return False
    return True

def binary_pauli_representation(commuting_pauli_set):
    Sz = commuting_pauli_set.z[:, ::-1].T
    Sx = commuting_pauli_set.x[:, ::-1].T
    return np.concatenate((Sz, Sx), axis=0).astype(int)

def mod2_gaussian_elimination(homogeneous_system_matrix):
    solution = homogeneous_system_matrix.copy().astype(int)
    nrows, ncols = homogeneous_system_matrix.shape
    h, k = 0, 0
    while (h < nrows) and (k < ncols):
        i_max = np.argmax(np.abs(solution[h:nrows, k])) + h
        if solution[i_max, k] == 0:
            k += 1
        else:
            solution[[h, i_max]] = solution[[i_max, h]] # Swap rows
            for i in range(h+1, nrows):
                f = solution[i, k]/solution[h, k]
                solution[i, k] = 0
                for j in range(k+1, ncols):
                    solution[i, j] = solution[i, j] - solution[h, j]*f
            h += 1
            k += 1
    return solution % 2

class Permutation:
    def __init__(self, perm):
        self.perm = np.array(perm)

    def __mul__(self, other):
        try:
            if len(self.perm) != len(other.perm):
                ValueError("Permutations must be the same size")
        except AttributeError:
            ValueError(f"{other} is not a permutation")

        composed_perm = self.perm[other.perm[np.arange(len(self.perm))]]
        return Permutation(composed_perm)

    @classmethod
    def transposition(cls, n, i, j):
        if (not (0 <= i < n)) or (not (0 <= j < n)):
            raise ValueError(f"i, j must be in range [0, n)")
        
        perm = np.arange(n)
        perm[[i, j]] = perm[[j, i]]
        return cls(perm)
    
    @classmethod
    def identity(cls, n):
        return cls(np.arange(n))
    
    def __call__(self, i):
        if not (0 <= i < len(self.perm)):
            raise ValueError("i must be in range [0, n)")
        return self.perm[i]

    def __getitem__(self, i):
        if not (0 <= i < len(self.perm)):
            raise ValueError("i must be in range [0, n)")
        return self.perm[i]
    
    def __repr__(self):
        return f"Transposition({self.perm})"

# INSTRUCTION ABSTRACTIONS

class H:
    def __init__(self, *qb):
        self.qubits = list(set(list(qb)))
        self._largestqb = max(self.qubits)

    def combine(self, other):
        if type(other) != H:
            raise ValueError("Only H can be combined")
        self.qubits = (self.qubits - other.qubits) | (other.qubits - self.qubits)
        self._largestqb = max(self.qubits) if len(self.qubits) > 0 else None

    def __repr__(self):
        indices = ", ".join([str(qb) for qb in self.qubits])
        return f"H({indices})"
    
    def binary_representation_conjugation(self, S):
        S = S.copy()
        S_qubits = S.shape[0] // 2
        if S_qubits < self._largestqb:
            raise ValueError(f"This gate acts on qubits not present in S")
        to_exchange = np.array(list(self.qubits)) + S_qubits
        S[[to_exchange, self.qubits]] = S[[self.qubits, to_exchange]]
        return S
    
    def phases_computation(self, phases, S, return_Sp=False):
        phases = phases.copy()
        nqubits = S.shape[0] // 2
        for qubit in self.qubits:
            z_row = S[qubit, :]
            x_row = S[nqubits+qubit, :]
            phases = (phases + z_row*x_row) % 2
        if return_Sp:
            return phases, self.binary_representation_conjugation(S)
        else:
            return phases

class P:
    def __init__(self, *qb):
        self.qubits = list(set(list(qb)))
        self._largestqb = max(self.qubits)
    
    def __repr__(self):
        indices = ", ".join([str(qb) for qb in self.qubits])
        return f"P({indices})"
    
    def binary_representation_conjugation(self, S):
        S = S.copy()
        S_qubits = S.shape[0] // 2
        if S_qubits < self._largestqb:
            raise ValueError(f"This gate acts on qubits not present in S")
        affected_rows = np.array(self.qubits)
        S[affected_rows] = (S[affected_rows] + S[affected_rows + S_qubits]) % 2
        return S
    
    def phases_computation(self, phases, S, return_Sp=False):
        phases = phases.copy()
        nqubits = S.shape[0] // 2
        for qubit in self.qubits:
            z_row = S[qubit, :]
            x_row = S[nqubits+qubit, :]
            phases = (phases + z_row*x_row) % 2
        if return_Sp:
            return phases, self.binary_representation_conjugation(S)
        else:
            return phases

class CZ:
    def __init__(self, ctrlqbs, tgqbs):
        try:
            clen = len(ctrlqbs)
            try:
                if len(tgqbs) != clen:
                    raise ValueError("Ctrlqbs and tgqbs must have the same length")
            except TypeError:
                raise ValueError("Ctrlqbs and tgqbs must have the same length")
        except TypeError:
                try:
                    len(tgqbs)
                    raise ValueError("Ctrlqbs and tgqbs must have the same length")
                except TypeError:
                    ctrlqbs = [ctrlqbs]
                    tgqbs = [tgqbs]

        for qb in ctrlqbs:
            if qb in tgqbs:
                raise ValueError("Ctrl and target qubits sets must not intersect")
        
        ctrl_counter = Counter(ctrlqbs)
        trgt_counter = Counter(tgqbs)
        for nctrl in ctrl_counter.values():
            if nctrl > 1:
                raise ValueError("Ther must not be repeated qubit indices in ctrlqbs")
        for ntrgt in trgt_counter.values():
            if ntrgt > 1:
                raise ValueError("Ther must not be repeated qubit indices in tgqbs")

        self._largestqb = max(max(ctrlqbs), max(tgqbs))
        self.qubits = [ctrlqbs, tgqbs]
    
    @property
    def ctrl(self):
        return self.qubits[0]

    @property
    def target(self):
        return self.qubits[1]
    
    @property
    def pairs(self):
        for ctrl, target in zip(self.ctrl, self.target):
            yield ctrl, target

    def binary_representation_conjugation(self, S):
        S = S.copy()
        S_qubits = S.shape[0] // 2
        if S_qubits < self._largestqb:
            raise ValueError(f"This gate acts on qubits not present in S")
        ctrls = np.array(self.ctrl)
        trgt = np.array(self.target)
        S[ctrls, :] = (S[ctrls, :] + S[trgt + S_qubits, :]) % 2
        S[trgt, :] = (S[trgt, :] + S[ctrls + S_qubits, :]) % 2
        return S
    
    def phases_computation(self, phases, S, return_Sp=False):
        phases = phases.copy()
        nqubits = S.shape[0] // 2
        for ctrlqb, targetqb in self.pairs:
            z_ctrl_row = S[ctrlqb, :]
            z_trgt_row = S[targetqb, :]
            x_ctrl_row = S[nqubits + ctrlqb, :]
            x_trgt_row = S[nqubits + targetqb, :]
            phases = (phases + x_ctrl_row*x_trgt_row*((z_ctrl_row + z_trgt_row) % 2)) % 2
        if return_Sp:
            return phases, self.binary_representation_conjugation(S)
        else:
            return phases

    def __repr__(self):
        indices = f"({", ".join([str(ctrlind) for ctrlind in self.ctrl])}), ({", ".join([str(trgtind) for trgtind in self.target])})"
        return f"CZ[{indices}]"

class InstructionList:
    def __init__(self, nqubits):
        self.nqubits = nqubits
        self.instructions = []

    def add(self, *instructions):
        try:
            for instruction in instructions:
                if instruction._largestqb >= self.nqubits:
                    raise ValueError("All instructions must fit in the InstructionList qubit set 0 <= n < N")
                if len(self.instructions) > 0:
                    if type(instruction) is type(self.instructions[-1]):
                        if type(instruction) is H:
                            self.instructions[-1].combine(instruction)
                            if len(self.instructions[-1].qubits) == 0:
                                del self.instructions[-1]
                        if type(instruction) is CZ:
                            last_pairs = list(self.instructions[-1].pairs)
                            pairs_to_add = list(instruction.pairs)
                            last_all_qb = set(self.instructions[-1].ctrl + self.instructions[-1].target)
                            for i, pair in enumerate(instruction.pairs):
                                if pair in last_pairs:
                                    last_pairs.remove(pair)
                                    pairs_to_add.remove(pair)
                                else: 
                                    if ({*pair} == ({*pair} - last_all_qb)):
                                        last_pairs.append(pair)
                                        pairs_to_add.remove(pair)
                            self.instructions[-1] = CZ([pair[0] for pair in last_pairs], [pair[1] for pair in last_pairs])
                            if len(pairs_to_add) > 0:
                                self.instructions.append(CZ([pair[0] for pair in pairs_to_add], [pair[1] for pair in pairs_to_add]))
                        if type(instruction) is P:
                            self.instructions.append(instruction)
                        else:
                            ValueError("Instructions must be a Clifford Object [H, P, CZ]")
                    else:
                        if type(instruction) in [H, P, CZ]:
                            self.instructions.append(instruction)
                        else:
                            ValueError("Instructions must be a Clifford Object [H, P, CZ]")
                else:
                    if type(instruction) in [H, P, CZ]:
                        self.instructions.append(instruction)
                    else:
                        ValueError("Instructions must be a Clifford Object [H, P, CZ]")
        except TypeError:
            if type(instruction) in [H, P, CZ]:
                self.instructions.append(instruction)
            else:
                ValueError("Instructions must be a Clifford Object [H, P, CZ]")
    
    def remove(self, index):
        del self.instructions[index]

    def __repr__(self):
        gates = ", ".join(str(instruction) for instruction in self.instructions)
        return f"InstructionList<{self.nqubits}>[{gates}]"
    
    def __getitem__(self, i):
        return self.instructions[i]

    def binary_representation_conjugation(self, S):
        for instruction in self.instructions:
            S = instruction.binary_representation_conjugation(S)
        return S
    
    def phases_computation(self, S, return_Sp=False):
        phases = np.zeros(S.shape[1], dtype=int)
        S = S.copy()
        for instruction in self.instructions:
            phases, S = instruction.phases_computation(phases, S, return_Sp=True)
        if return_Sp:
            return phases, S
        else:
            return phases
    
    def to_qiskit_circuit(self, barriers_before_cz=True):
        qc = QuantumCircuit(self.nqubits)
        last_instruction_type = None
        for instruction in self.instructions:
            if last_instruction_type != CZ and type(instruction) == CZ:
                qc.barrier()
            if type(instruction) == H:
                qc.h(instruction.qubits)
                last_instruction_type = H
            elif type(instruction) == P:
                qc.s(instruction.qubits)
                last_instruction_type = P
            else:
                qc.cz(instruction.ctrl, instruction.target)
                last_instruction_type = CZ
        return simplify_logical_circuits(qc)
    
# UNITARY SYNTHESIS

# References: Quantum 5, 385 (2021)
#             PRA 70, 052328 (2004) Lemma 3

def get_S_Rinv(pauli_representation):
    gauss_pauli_representation = mod2_gaussian_elimination(pauli_representation)
    nrows, ncols = gauss_pauli_representation.shape
    pivot_cols = []
    for j in range(ncols):
        if np.sum(gauss_pauli_representation[j:, j].astype(int)) == 1:
            pivot_cols.append(j)
    S = pauli_representation[:, pivot_cols]
    R0_inv = gauss_pauli_representation[pivot_cols]
    return S, R0_inv

def first_reduction_step(S):
    # PRA 70, 052328 (2004) Lemma 3 (Adapted to the other's article notation)
    nrows = S.shape[0]//2
    ncols = S.shape[1]
    S_gauss = mod2_gaussian_elimination(S.T).T
    S_rank = np.linalg.matrix_rank(S_gauss)
    Sx_rank = np.linalg.matrix_rank(S_gauss[nrows:])
    C_diag = np.diagonal(S_gauss[:S_rank-Sx_rank, :S_rank-Sx_rank])
    # zero_diag_elements = np.where(np.equal(C_diag, 0))[0]
    row_swaps = Permutation.identity(nrows)
    for i in range(len(C_diag)):
        if C_diag[i] == 0:
            zero_diag_col = S[:, i]
            first_one_row_ind = np.where(np.equal(zero_diag_col, 1))[0][0]
            row_swaps = row_swaps * Permutation.transposition(nrows, i, first_one_row_ind)
            S_gauss[[i, first_one_row_ind]] = S[[first_one_row_ind, i]]
    hadamard_qubits = []
    for i, row in enumerate(S_gauss[nrows:, :]):
        if row[::-1][min(i, ncols - 1)] != 1:
            hadamard_qubits.append(row_swaps(i))
    instructions = InstructionList(nrows)
    instructions.add(H(*hadamard_qubits))
    S1 = instructions.binary_representation_conjugation(S)
    return S1, instructions

def second_reduction_step(S1, instructions=None):
    nrows = S1.shape[0]//2
    if instructions is None:
        instructions = InstructionList(nrows)
    R1_inv = S1[nrows:nrows+np.linalg.matrix_rank(S1), :] % 2
    R1 = (np.linalg.inv(R1_inv).astype(int))
    S2 = (S1 @ R1) % 2
    return S2, R1_inv, instructions

def third_reduction_step(S2, instructions=None):
    nrows = S2.shape[0]//2
    initial_ncols = S2.shape[1]
    ncols = initial_ncols
    if instructions is None:
        instructions = InstructionList(nrows)
    symbols = [sym.symbols(f"b{i}") for i in range(2*nrows)]
    reversed_blocks_symbols = symbols[nrows:] + symbols[:nrows]
    S2_rank = np.linalg.matrix_rank(S2)
    S3 = S2.copy()
    for n in range(S2_rank+1, nrows+1):
        expressions = []
        for colind in range(ncols):
            col = S3[:, colind]
            nonzero_indices = np.where(col)[0]
            if len(nonzero_indices) > 1:
                this_expression = reversed_blocks_symbols[nonzero_indices[0]] ^ reversed_blocks_symbols[nonzero_indices[1]]
                for nonzero_index in nonzero_indices[2::]:
                    this_expression = this_expression ^ reversed_blocks_symbols[nonzero_index]
            else:
                this_expression = reversed_blocks_symbols[nonzero_indices[0]]
            expressions.append(~this_expression)
        last_expression = symbols[0] & symbols[nrows]
        for i in range(1, nrows):
            last_expression = (last_expression) ^ (symbols[i] & symbols[nrows + i])
        expressions.append(~last_expression)
        final_clause = expressions[0] & expressions[1]
        for expression in expressions[2::]:
            final_clause = final_clause & expression
        clause_valid_solutions = sym.logic.inference.satisfiable(final_clause, all_models=True, algorithm="z3") # Z3 SAT algorithm prioritizes low weight solutions
        for valid_solution in clause_valid_solutions:
            solution_col = np.array([valid_solution[var] for var in symbols], dtype=bool)
            if (solution_col[nrows:].sum() == 0):
                continue
            S3_candidate = np.concatenate((S3, solution_col.reshape(2*nrows, 1)), axis=1)
            if np.linalg.matrix_rank(S3_candidate[nrows:]) == n:
                S3 = S3_candidate.copy()
                ncols = S3.shape[1]
                break
    R2_inv = np.concatenate((np.identity(initial_ncols, dtype=int), np.zeros((ncols - initial_ncols, initial_ncols), dtype=int)), axis=0)
    return S3, R2_inv, instructions

def fourth_reduction_step(S3, S2, instructions=None):
    nrows = S3.shape[0]//2
    if instructions is None:
        instructions = InstructionList(nrows)
    R3_inv = S3[nrows:, :]
    R3 = np.linalg.inv(R3_inv)
    S4prev = ((S3 @ R3) % 2)
    S2rank = np.linalg.matrix_rank(S2)
    Ediag = np.diagonal(S4prev)
    non_zero_indices = np.where(Ediag)[0]
    if len(non_zero_indices) > 0:
        Q = P(*non_zero_indices)
        instructions.add(Q)
        S4 = Q.binary_representation_conjugation(S4prev)
        return S4, R3_inv, instructions
    else:
        return S4prev, R3_inv, instructions

def fifth_reduction_step(S4, instructions=None):
    nrows = S4.shape[0]//2
    if instructions is None:
        instructions = InstructionList(nrows)
    S5 = S4.copy()
    upper_half = S5[:nrows, :]
    lower_half = S5[nrows:, :]
    # Reduce upper half
    non_zeros_upper = np.where(upper_half)
    cz_pairs = np.concatenate((non_zeros_upper[0].reshape(len(non_zeros_upper[0]), 1), non_zeros_upper[1].reshape(len(non_zeros_upper[1]), 1)), axis=1)
    cz_pairs = np.unique(np.sort(cz_pairs, axis=1), axis=0)
    cz_pairs_distance = np.abs(cz_pairs[:, 1] - cz_pairs[:, 0])
    mask = np.argsort(cz_pairs_distance)
    cz_pairs = cz_pairs[mask, :]
    Q = InstructionList(nrows)
    for pair in cz_pairs:
        this_Q = CZ(pair[0], pair[1])
        Q.add(this_Q)
    S5 = Q.binary_representation_conjugation(S5)
    instructions.add(*Q)
    # Add Hadamards to flip blocks
    last_hadamards = H(*np.arange(nrows))
    S5 = last_hadamards.binary_representation_conjugation(S5)
    instructions.add(last_hadamards)
    return S5, instructions

def reduce_binary_string_representation(pauli_representation):
    S, R0_inv = get_S_Rinv(pauli_representation)
    S1, instructions = first_reduction_step(S)
    S2, R1_inv, instructions = second_reduction_step(S1, instructions)
    S3, R2_inv, instructions = third_reduction_step(S2, instructions)
    S4, R3_inv, instructions = fourth_reduction_step(S3, S2, instructions)
    S5, instructions = fifth_reduction_step(S4, instructions)
    R_inv = (R3_inv @ R2_inv @ R1_inv @ R0_inv) % 2
    phases = instructions.phases_computation(pauli_representation)
    return S5, instructions, R_inv, phases

def paulis_diagonalization_circuit(commuting_pauli_list):
    # Quantum 5, 385 (2021)
    all_commute = len(commuting_pauli_list.commutes_with_all(commuting_pauli_list)) == len(commuting_pauli_list)
    if not all_commute:
        raise ValueError("The provided PauliList contains non-commuting operators")
    nqubits = len(commuting_pauli_list[0])
    non_diagonal_qubits_num = count_non_diagonal_qubits(commuting_pauli_list)
    if non_diagonal_qubits_num == 0:
        return QuantumCircuit(nqubits)
    Sp = binary_pauli_representation(commuting_pauli_list)
    redS, instructions, R_inv, phases = reduce_binary_string_representation(Sp)
    return instructions.to_qiskit_circuit(), R_inv, phases

# -----------------------------------------
#     CIRCUIT EXECUTION AND MEASUREMENT
# -----------------------------------------

def load_job_result(filepath):
    with open(filepath, "r") as f:
        result = json.load(f, cls=RuntimeDecoder)
    return result

def save_job_result(filepath, job):
    result = job.result()
    with open(filepath, "w") as f:
        json.dump(result, f, cls=RuntimeEncoder)
    return result

def get_samples_layout_map(physical_circ_layout):
    final_indices = np.arange(len(physical_circ_layout))[np.argsort(physical_circ_layout)]
    return final_indices
    
def get_layout_state(state_arr_str, physical_circ_layout):
    state_arr_reordering = np.array(physical_circ_layout).argsort().argsort()
    if type(state_arr_str) == str:
        return "".join(np.array(list(state_arr_str[::-1]))[state_arr_reordering])
    else:
        return state_arr_str[..., ::-1][..., state_arr_reordering]
    
def undo_layout_state(layout_state_str, physical_circ_layout):
    state_arr_reordering = np.array(physical_circ_layout).argsort().argsort()
    state_arr_reordering_inv = np.zeros(len(physical_circ_layout), dtype=int)
    for i, value in enumerate(state_arr_reordering):
        state_arr_reordering_inv[value] = i
    if type(layout_state_str) == str:
        return "".join(np.array(list(layout_state_str))[state_arr_reordering_inv][::-1])
    else:
        return layout_state_str[..., state_arr_reordering_inv][..., ::-1]

def diagonal_operators_check(operators):
    different_paulis = set("".join([pauli_to_str(op) for op in operators]))
    if len(remaining_ops := (different_paulis - {"I"})) > 1:
        return False
    basis = remaining_ops.pop()
    all_diagonal = all([is_diagonal(op, basis=basis) for op in operators])
    return all_diagonal, basis

def initialize_postselection(nqubits, postselects_generating_func, observable_generating_funcs):
    if type(observable_generating_funcs) != list:
        observable_generating_funcs = [observable_generating_funcs]

    postselection_ops = PauliList(set(postselects_generating_func(nqubits)))
    post_selection_diagonal, basis = diagonal_operators_check(postselection_ops)
    if not post_selection_diagonal:
        raise NotImplementedError("Only supports diagonal postselection operators in some basis")
    logical_observables = PauliList([Pauli(observable_func(nqubits)) for observable_func in observable_generating_funcs])
    all_commute = check_postselection_observable_commutation(logical_observables, postselection_ops)
    if not all_commute:
        raise ValueError("All observables and postselection operators must commute")
    diagonal_observables = []
    non_diagonal_observables = []
    for observable in logical_observables:
        if is_diagonal(observable, basis=basis):
            diagonal_observables.append(observable)
        else:
            non_diagonal_observables.append(observable)
    return postselection_ops, diagonal_observables, non_diagonal_observables, basis

def execute_postselected_sampler_batch(backend, sampler_opt_dict, transpiled_circuits, postselects_generating_func, observable_generating_funcs, extra_options=None, job_db=None):
    # TODO: This only works for our particular case
    nqubits = np.max([count_non_idle_qubits(circ) for circ in transpiled_circuits])
    postselection_ops, diagonal_observables, non_diagonal_observables, basis = initialize_postselection(nqubits, postselects_generating_func, observable_generating_funcs)
    if basis != "Z":
        basis_changed_circuits = [append_basis_change_circuit(circ, basis, backend) for circ in transpiled_circuits]
    else:
        basis_changed_circuits = transpiled_circuits
    # First send diagonal jobs
    jobs = execute_sampler_batch(backend, sampler_opt_dict, basis_changed_circuits)
    # Diagonalize each non-diagonal observable and send a batch of jobs to avoid long circuits
    for observable in non_diagonal_observables:
        raise NotImplementedError("Only supports diagonal observables")
        diag_range = diagonalization_susceptible_qb_range([observable], postselection_ops)
        reduced_ops_to_diag = PauliList([op[diag_range[0]:diag_range[1]+1] for op in postselection_ops + PauliList([observable])])
        this_diag_circ, this_R_inv, this_phases = paulis_diagonalization_circuit(reduced_ops_to_diag)
            
    if job_db is not None:
        job_ids = [job.job_id() for job in jobs]
        postselection_strings = sorted([pauli_to_str(op) for op in postselection_ops])
        observables_string = sorted([pauli_to_str(op) for op in set(diagonal_observables + non_diagonal_observables)])
        post_obs_info_dict = {"postselection_ops": postselection_strings, "observables": observables_string}
        extra_options = {} if extra_options is None else extra_options
        options = extra_options | post_obs_info_dict | sampler_opt_dict
        job_db.add(options, transpiled_circuits, "Sampler", job_ids)
    else:
        print("WARNING: Submitting jobs without job_db")
    
    return jobs

def db_add_postselected_sampler_ran_batch(job_db, service, session_id, backend, sampler_opt_dict, transpiled_circuits, postselects_generating_func, observable_generating_funcs, extra_options=None):
    nqubits = np.max([count_non_idle_qubits(circ) for circ in transpiled_circuits])
    postselection_ops, diagonal_observables, non_diagonal_observables, basis = initialize_postselection(nqubits, postselects_generating_func, observable_generating_funcs)
    if basis != "Z":
        basis_changed_circuits = [append_basis_change_circuit(circ, basis, backend) for circ in transpiled_circuits]
    else:
        basis_changed_circuits = transpiled_circuits
    
    jobs = service.jobs(session_id=session_id, limit=len(transpiled_circuits)*(len(non_diagonal_observables)+1))[::-1]
    job_ids = [job.job_id() for job in jobs]
    postselection_strings = sorted([pauli_to_str(op) for op in postselection_ops])
    observables_string = sorted([pauli_to_str(op) for op in set(diagonal_observables + non_diagonal_observables)])
    post_obs_info_dict = {"postselection_ops": postselection_strings, "observables": observables_string}
    extra_options = {} if extra_options is None else extra_options
    options = extra_options | post_obs_info_dict | sampler_opt_dict
    job_db.add(options, transpiled_circuits, "Sampler", job_ids)

    return jobs

def is_valid_state_string(string, postselection_ops):
    nqubits = len(postselection_ops[0])
    for operator in postselection_ops:
        if len(operator) != nqubits:
            raise ValueError("All operators must act on the same number of qubits")
    if len(string) != nqubits:
        raise ValueError("The state string and all postselection operators must have the same number of qubits")
    all_diagonal, basis = diagonal_operators_check(postselection_ops)
    if not all_diagonal:
        raise ValueError("Only supports diagonal postselection operators in some basis")
    postselection_mask = np.array([[int(str(opel) == basis) for opel in operator] for operator in postselection_ops])
    string_arr = np.array([int(c) for c in string])
    string_rep_arr = np.repeat(np.array([string_arr]), len(postselection_ops), axis=0)
    each_postselection_valid = ~(np.sum(string_rep_arr*postselection_mask, axis=1) % 2).astype(bool)
    return np.all(each_postselection_valid)

def get_postselected_samples_dict(samples_dict, postselection_ops, circ_layout=None):
    all_diagonal, basis = diagonal_operators_check(postselection_ops)
    if not all_diagonal:
        raise ValueError("Only supports diagonal postselection operators in some basis")
    try:
        strings_arr = np.array([[int(c) for c in string] for string in samples_dict.keys()])
        postselection_mask = np.array([[int(str(opel) == basis) for opel in operator] for operator in postselection_ops])
    except ValueError:
        raise ValueError("All postselection operators and samples must have the same number of qubits")
    rep_strings_arr = np.repeat(strings_arr, len(postselection_ops), axis=0)
    rep_postselection_mask = np.tile(postselection_mask.T, len(samples_dict)).T
    if circ_layout is not None:
        rep_strings_arr = get_layout_state(rep_strings_arr, circ_layout)
    each_string_postselections = ~(np.sum(rep_strings_arr*rep_postselection_mask, axis=1) % 2).reshape(len(samples_dict), len(postselection_ops)).astype(bool)
    is_valid_string = np.all(each_string_postselections, axis=1)
    return {state_string:counts for i, (state_string, counts) in enumerate(samples_dict.items()) if is_valid_string[i]}

def get_recovered_postselected_samples_dict(samples_dict, postselection_ops, circ_layout=None):
    all_diagonal, basis = diagonal_operators_check(postselection_ops)
    if not all_diagonal:
        raise ValueError("Only supports diagonal postselection operators in some basis")
    try:
        strings_arr = np.array([[int(c) for c in string] for string in samples_dict.keys()])
        postselection_mask = np.array([[int(str(opel) == basis) for opel in operator] for operator in postselection_ops])
    except ValueError:
        raise ValueError("All postselection operators and samples must have the same number of qubits")
    decoder = Matching(postselection_mask)
    if circ_layout is not None:
        strings_arr = get_layout_state(strings_arr, circ_layout)
    string_syndromes = (strings_arr @ postselection_mask.T) % 2
    predicted_flips = decoder.decode_batch(string_syndromes)
    recovered_strings = (strings_arr + predicted_flips) % 2
    if circ_layout is not None:
        recovered_strings = undo_layout_state(recovered_strings, circ_layout)
    recovered_samples_dict = {}
    fmt_str = "%i"*recovered_strings.shape[1]
    for i, string in enumerate(samples_dict.keys()):
        recovered_string = fmt_str % tuple(recovered_strings[i])
        if recovered_samples_dict.get(recovered_string, None) is not None:
            recovered_samples_dict[recovered_string] = recovered_samples_dict[recovered_string] + samples_dict[string]
        else:
            recovered_samples_dict[recovered_string] = samples_dict[string]
    return recovered_samples_dict

def measure_diagonal_observables(samples_dict, diagonal_observables, circ_layout=None):
    nqubits = len(diagonal_observables[0])
    for observable in diagonal_observables:
        if len(observable) != nqubits:
            raise ValueError("All observables must act on the same number of qubits")
    for state_string in samples_dict.keys():
        if len(state_string) != nqubits:
            raise ValueError("The state strings and all observables must act on the same number of qubits")
    all_diagonal, basis = diagonal_operators_check(diagonal_observables)
    if not all_diagonal:
        raise ValueError("The provided observables are not diagonal in the same basis")
    strings_expectation_values = np.zeros((len(samples_dict), len(diagonal_observables)))
    observables_mask = np.array([[int(str(opel) == basis) for opel in observable] for observable in diagonal_observables])
    for i, state_string in enumerate(samples_dict.keys()):
        state_string_rep = np.repeat(np.array([[int(c) for c in state_string]]), [len(diagonal_observables)], axis=0)
        if circ_layout is not None:
            state_string_rep = get_layout_state(state_string_rep, circ_layout)
        observables_expectation_value = (-1)**(np.sum(state_string_rep*observables_mask, axis=1) % 2)
        strings_expectation_values[i] = observables_expectation_value
    weights = np.array(list(samples_dict.values()))/np.sum(list(samples_dict.values()))
    strings_expectation_times_weights = strings_expectation_values * weights.reshape((len(samples_dict), 1))
    expectation_values = np.sum(strings_expectation_times_weights, axis=0)
    return expectation_values

def simulate_postselected_operators(fake_backend, sampler_opt_dict, transpiled_circuits, postselect_generating_func, observable_generating_funcs, return_samples_dicts=False, return_postselected_samples_dicts=False):
    nqubits = np.max([count_non_idle_qubits(circ) for circ in transpiled_circuits])
    postselection_ops, diagonal_observables, non_diagonal_observables, basis = initialize_postselection(nqubits, postselect_generating_func, observable_generating_funcs)
    # Express diagonal observables in the computational basis
    if basis != "Z":
        basis_changed_circuits = [append_basis_change_circuit(circ, basis, fake_backend) for circ in transpiled_circuits]
    else:
        basis_changed_circuits = transpiled_circuits
    site_gauge_observable_matrix = np.zeros((len(basis_changed_circuits), len(diagonal_observables + non_diagonal_observables)))
    # Execute diagonal circuits
    jobs = execute_sampler_batch(fake_backend, sampler_opt_dict, basis_changed_circuits)
    # Execute non diagonal circuits
    non_diagonal_obs_R_invs = []
    non_diagonal_obs_phases = []
    affected_postselection_operators = np.zeros((len(non_diagonal_observables), len(postselection_ops)), dtype=bool)
    for i, observable in enumerate(non_diagonal_observables):
        diag_range = diagonalization_susceptible_qb_range([observable], postselection_ops)
        reduced_pops_to_diag = []
        for j, op in enumerate(postselection_ops):
            this_reduced_op = op[diag_range[0]:diag_range[1]+1]
            if len(this_reduced_op) != pauli_to_str(this_reduced_op).count("I"):
                reduced_pops_to_diag.append(this_reduced_op)
                affected_postselection_operators[i, j] = True
        reduced_ops_to_diag = PauliList(reduced_pops_to_diag + [observable])
        this_diag_circ, this_R_inv, this_phases = paulis_diagonalization_circuit(reduced_ops_to_diag)
        this_obs_circs = []
        for circ in transpiled_circuits:
            this_circ_layout = circ.layout.final_index_layout()[diag_range[0]:diag_range[1]+1]
            pm = generate_preset_pass_manager(backend=fake_backend, optimization_level=2, initial_layout=this_circ_layout)
            transpiled_diag_circ = pm.run(this_diag_circ)
            this_obs_circs.append(join_transpiled_circuits(circ, transpiled_diag_circ, backend=fake_backend))
        non_diagonal_obs_R_invs.append(this_R_inv)
        non_diagonal_obs_phases.append(this_phases)
        this_obs_jobs = execute_sampler_batch(fake_backend, sampler_opt_dict, this_obs_circs)
        jobs = jobs + this_obs_jobs
    if return_samples_dicts: samples_dicts = []
    if return_postselected_samples_dicts: postselected_samples_dicts = []
    # Measure diagonal postselected operators
    diagonal_jobs = jobs[:len(basis_changed_circuits)]
    for i, djob in enumerate(diagonal_jobs):
        samples_dict = list(djob.result()[0].data.values())[0].get_counts()
        if return_samples_dicts: samples_dicts.append(samples_dict)
        postselected_samples_dict = get_postselected_samples_dict(samples_dict, postselection_ops, circ_layout=basis_changed_circuits[i].layout.final_index_layout())
        if return_postselected_samples_dicts: postselected_samples_dicts.append(postselected_samples_dict)
        expectation_values = measure_diagonal_observables(samples_dict, diagonal_observables, circ_layout=basis_changed_circuits[i].layout.final_index_layout())
        site_gauge_observable_matrix[i, :len(diagonal_observables)] = expectation_values
    # TODO: Measure non-diagonal observables
    for i, observable in enumerate(non_diagonal_observables):
        this_observable_jobs = jobs[(i+1)*len(basis_changed_circuits):(i+2)*len(basis_changed_circuits)]
        this_postselection_operators = None
    to_return = [site_gauge_observable_matrix]
    if return_samples_dicts:
        to_return.append(samples_dicts)
    if return_postselected_samples_dicts:
        to_return.append(postselected_samples_dicts)
    return to_return if len(to_return) > 1 else to_return[0]

def load_postselected_jobs(job_db, ibmq_service, sampler_opt_dict, transpiled_circuits, postselect_generating_func, observable_generating_funcs, extra_options=None, job_index=0, jobs_result_folder="", return_samples_dicts=False, return_postselected_samples_dicts=False):
    nqubits = np.max([count_non_idle_qubits(circ) for circ in transpiled_circuits])
    postselection_ops, diagonal_observables, non_diagonal_observables, basis = initialize_postselection(nqubits, postselect_generating_func, observable_generating_funcs)
    postselection_strings = sorted([pauli_to_str(op) for op in postselection_ops])
    observables_string = sorted([pauli_to_str(op) for op in PauliList(set(diagonal_observables + non_diagonal_observables))])
    post_obs_info_dict = {"postselection_ops": postselection_strings, "observables": observables_string}
    extra_options = {} if extra_options is None else extra_options
    options = extra_options | post_obs_info_dict | sampler_opt_dict
    jobs_layout = []
    if os.path.isdir(jobs_result_folder):
        jobs_db_json = job_db.search_by_params(options, transpiled_circuits, "Sampler", strict_depth=False, limit=job_index+1)
        jobs = []
        for job_json in jobs_db_json["jobs"]:
            this_job_id = job_json["job_id"]
            if os.path.isfile(this_job_filepath := os.path.join(jobs_result_folder, f"{this_job_id}.json")):
                this_result = load_job_result(this_job_filepath)
                jobs.append(this_result)
            else:
                this_job_object = ibmq_service.job(job_id=this_job_id)
                this_result = save_job_result(this_job_filepath, this_job_object)
                jobs.append(this_result)
            jobs_layout.append(job_json["layout"])
    else:
        jobs_db_json = job_db.search_by_params(options, transpiled_circuits, "Sampler", strict_depth=False, limit=job_index+1)
        jobs = []
        jobs_layout = []
        for job_json in jobs_db_json["jobs"]:
            jobs.append(ibmq_service.job(job_id=job_json["job_id"]))
            jobs_layout.append(job_json["layout"])
    site_gauge_observable_matrix = np.zeros((len(transpiled_circuits), len(diagonal_observables + non_diagonal_observables)))
    if return_samples_dicts: samples_dicts = []
    if return_postselected_samples_dicts: postselected_samples_dicts = []
    # Measure diagonal postselected operators
    diagonal_jobs = jobs[:len(transpiled_circuits)]
    for i, djob in enumerate(diagonal_jobs):
        if os.path.isdir(jobs_result_folder):
            samples_dict = list(djob[0].data.values())[0].get_counts()
        else:
            samples_dict = list(djob.result()[0].data.values())[0].get_counts()
        if return_samples_dicts: samples_dicts.append(samples_dict)
        # postselected_samples_dict = get_postselected_samples_dict(samples_dict, postselection_ops, jobs_layout[i])
        postselected_samples_dict = get_recovered_postselected_samples_dict(samples_dict, postselection_ops, jobs_layout[i])
        if return_postselected_samples_dicts: postselected_samples_dicts.append(postselected_samples_dict)
        expectation_values = measure_diagonal_observables(postselected_samples_dict, diagonal_observables, jobs_layout[i])
        site_gauge_observable_matrix[i, :len(diagonal_observables)] = expectation_values
    # Measure non-diagonal observables
    for observable in non_diagonal_observables:
        raise NotImplementedError("Non-diagonal observables not yet supported")
    to_return = [site_gauge_observable_matrix]
    if return_samples_dicts:
        to_return.append(samples_dicts)
    if return_postselected_samples_dicts:
        to_return.append(postselected_samples_dicts)
    return to_return if len(to_return) > 1 else to_return[0]