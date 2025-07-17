from utils.circs import simplify_logical_circuits, count_non_idle_qubits, append_basis_change_circuit, join_transpiled_circuits, numpy_array_of_circuits
from qiskit_ibm_runtime.exceptions import RuntimeJobFailureError, RuntimeInvalidStateError
from utils.odr import compute_depolarizing_probability, renormalize_observable
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from utils.postselection_weights import decode_with_optimized_weights
from qiskit_ibm_runtime import RuntimeDecoder, RuntimeEncoder
from utils.bootstrap import bootstrap_sample_dict_mean
from qiskit.quantum_info import Pauli, PauliList
from utils.hexec import execute_sampler_batch
from qiskit_ibm_runtime import Batch
from qiskit import QuantumCircuit
from collections import Counter
from pymatching import Matching
from itertools import product
import multiprocessing as mp
import utils.dec_opt as dec
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
        ctrl_indices = ", ".join([str(ctrlind) for ctrlind in self.ctrl])
        target_indices = ", ".join([str(trgtind) for trgtind in self.target])
        indices = f"({ctrl_indices}), ({target_indices})"
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

def combine_samples_dicts(*samples_dicts):
    final_samples_dict = samples_dicts[0].copy()
    for samples_dict in samples_dicts[1::]:
        for state in samples_dict.keys():
            if final_samples_dict.get(state, None) is not None:
                final_samples_dict[state] += samples_dict[state]
            else:
                final_samples_dict[state] = samples_dict[state]
    return final_samples_dict

def generate_observables(nqubits, observable_generating_funcs, diagonal_basis="Z"):
    if not isinstance(observable_generating_funcs, (tuple, list)):
        observable_generating_funcs = [observable_generating_funcs]
    logical_observables = PauliList([Pauli(observable_func(nqubits)) for observable_func in observable_generating_funcs])
    diagonal_observables = []
    non_diagonal_observables = []
    for observable in logical_observables:
        if is_diagonal(observable, basis=diagonal_basis):
            diagonal_observables.append(observable)
        else:
            non_diagonal_observables.append(observable)
    return diagonal_observables, non_diagonal_observables

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

def generate_multirep_circuits_to_run(transpiled_circuits, cal_circuits, nrepetitions):
    transpiled_circuits = numpy_array_of_circuits(transpiled_circuits)
    cal_circuits = numpy_array_of_circuits(cal_circuits)

    if transpiled_circuits.shape != cal_circuits.shape:
        raise ValueError("If provided, cal_circuits must have the same shape as transpiled_circuits")

    if len(transpiled_circuits.shape) < 2:
        transpiled_circuits = transpiled_circuits[None]
        if cal_circuits is not None:
            cal_circuits = np.array(cal_circuits)[None]

    circuits_to_run = []
    for ilayout in range(transpiled_circuits.shape[0]):
        # this_layout_circuits = np.concatenate((cal_circuits[ilayout], np.tile(transpiled_circuits[ilayout], nrepetitions)))
        this_layout_circuits = np.column_stack([cal_circuits[ilayout]] + [transpiled_circuits[ilayout]]*nrepetitions).flatten()
        circuits_to_run.append(this_layout_circuits)
    return np.array(circuits_to_run)

def execute_postselected_sampler_batch(backend, sampler_opt_dict, transpiled_circuits, observable_generating_funcs, postselects_generating_func=None, cal_circuits=None, extra_options=None, job_db=None, max_shots=75000, diagonal_basis="Z"):
    # TODO: This only works for our particular case
    transpiled_circuits = numpy_array_of_circuits(transpiled_circuits)
    if cal_circuits is not None: cal_circuits = numpy_array_of_circuits(cal_circuits)
    if len(transpiled_circuits.shape) > 1:
        nqubits = 0
        for circ_arr in transpiled_circuits:
            for circ in circ_arr:
                if (this_nqb := count_non_idle_qubits(circ)) > nqubits:
                    nqubits = this_nqb
    else:
        nqubits = np.max([count_non_idle_qubits(circ) for circ in transpiled_circuits])
    
    if postselects_generating_func is None:
        diagonal_observables, non_diagonal_observables = generate_observables(nqubits, observable_generating_funcs, diagonal_basis=diagonal_basis)
    else:
        postselection_ops, diagonal_observables, non_diagonal_observables, diagonal_basis = initialize_postselection(nqubits, postselects_generating_func, observable_generating_funcs)
    if diagonal_basis != "Z":
        transpiled_circuits = np.array([append_basis_change_circuit(circ, diagonal_basis, backend) for circ in transpiled_circuits])
    if cal_circuits is not None:
        try:
            if cal_circuits.shape != transpiled_circuits.shape:
                raise ValueError("If provided, odr_circuits must be the same length as transpiled_circuits")
        except TypeError:
            raise ValueError("If provided, odr_circuits must be the same length as transpiled_circuits")

    if len(transpiled_circuits.shape) > 1:
        shots = sampler_opt_dict.get("default_shots", 1)
        shots_per_layout = shots // transpiled_circuits.shape[0]
    else:
        shots = sampler_opt_dict.get("default_shots", 1)
        shots_per_layout = shots
    
    if shots_per_layout > max_shots:
        nrepetitions = shots_per_layout // max_shots
        if nrepetitions == 0:
            nrepetitions = 1
        last_rep_shots = shots % max_shots
        sampler_options_per_sim = []
        for i in range(nrepetitions+1):
            this_options = sampler_opt_dict.copy()
            if i == nrepetitions:
                this_options["default_shots"] = last_rep_shots if last_rep_shots != 0 else max_shots
            else:
                this_options["default_shots"] = max_shots
            sampler_options_per_sim.append(this_options)
        # if last_rep_shots == 0 and len(sampler_options_per_sim) != nrepetitions:
        #     sampler_options_per_sim.pop()
    else:
        nrepetitions = 1
        edited_sampler_options = sampler_opt_dict.copy()
        edited_sampler_options["default_shots"] = shots_per_layout
        sampler_options_per_sim = [edited_sampler_options]*2
    
    if len(transpiled_circuits.shape) < 2:
        transpiled_circuits = transpiled_circuits[None]
        if cal_circuits is not None:
            cal_circuits = np.array(cal_circuits)[None]

    circuits_to_run = generate_multirep_circuits_to_run(transpiled_circuits, cal_circuits, nrepetitions)

    batch = Batch(backend=backend)

    # First send diagonal jobs
    job_objs = []
    for i, this_layout_circs in enumerate(circuits_to_run):
        # this_jobs = execute_sampler_batch(backend, sampler_options_per_layout, transpiled_circuits[i], extra_circuits=cal_circuits[i] if cal_circuits is not None else None, batch=batch)
        this_jobs = execute_sampler_batch(backend, sampler_options_per_sim*transpiled_circuits.shape[1], this_layout_circs, batch=batch)
        job_objs.append(this_jobs)
    job_objs = np.array(job_objs)

    batch.close()

    # Diagonalize each non-diagonal observable and send a batch of jobs to avoid long circuits
    for observable in non_diagonal_observables:
        raise NotImplementedError("Only supports diagonal observables")
        diag_range = diagonalization_susceptible_qb_range([observable], postselection_ops)
        reduced_ops_to_diag = PauliList([op[diag_range[0]:diag_range[1]+1] for op in postselection_ops + PauliList([observable])])
        this_diag_circ, this_R_inv, this_phases = paulis_diagonalization_circuit(reduced_ops_to_diag)

    if job_db is not None:
        job_ids = np.array([[job.job_id() for job in this_layout_jobs] for this_layout_jobs in job_objs])
        postselection_strings = sorted([pauli_to_str(op) for op in postselection_ops])
        observables_string = sorted([pauli_to_str(op) for op in set(diagonal_observables + non_diagonal_observables)])
        if postselects_generating_func is not None:
            post_obs_info_dict = {"postselection_ops": postselection_strings, "observables": observables_string}
        else:
            post_obs_info_dict = {}
        extra_options = {} if extra_options is None else extra_options
        options = extra_options | sampler_opt_dict | ({"cal": True} if cal_circuits is not None else {"cal": False}) | post_obs_info_dict
        job_db.add(options, circuits_to_run.flatten().tolist(), "Sampler", job_ids.flatten().tolist())
    else:
        print("WARNING: Submitting jobs without job_db")
    
    return job_objs.flatten().tolist()
    
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

def get_postselected_samples_dict(samples_dict, postselection_ops, circ_layout=None, return_logical=False):
    if len(samples_dict) == 0:
        return samples_dict
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
    if circ_layout is not None and not return_logical:
        rep_strings_arr = get_layout_state(rep_strings_arr, circ_layout)
    each_string_postselections = ~(np.sum(rep_strings_arr*rep_postselection_mask, axis=1) % 2).reshape(len(samples_dict), len(postselection_ops)).astype(bool)
    is_valid_string = np.all(each_string_postselections, axis=1)
    return {state_string:counts for i, (state_string, counts) in enumerate(samples_dict.items()) if is_valid_string[i]}

def get_recovered_postselected_samples_dict(samples_dict, postselection_ops, circ_layout=None, flip_threshold=None, return_logical=False):
    if len(samples_dict) == 0:
        return samples_dict
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
    npredicted_flips = predicted_flips.sum(axis=1)
    recovered_strings = (strings_arr + predicted_flips) % 2
    if circ_layout is not None and not return_logical:
        recovered_strings = undo_layout_state(recovered_strings, circ_layout)
    recovered_samples_dict = {}
    fmt_str = "%i"*recovered_strings.shape[1]
    for i, string in enumerate(samples_dict.keys()):
        recovered_string = fmt_str % tuple(recovered_strings[i])
        if flip_threshold is not None:
            if npredicted_flips[i] > flip_threshold:
                continue
        if recovered_samples_dict.get(recovered_string, None) is not None:
            recovered_samples_dict[recovered_string] = recovered_samples_dict[recovered_string] + samples_dict[string]
        else:
            recovered_samples_dict[recovered_string] = samples_dict[string]
    return recovered_samples_dict

def get_logical_samples_dict(samples_dict, circ_layout):
    new_layout_dict = {}
    strings_arr = np.array([[int(c) for c in state_string] for state_string in samples_dict.keys()])
    if len(strings_arr) > 0:
        if strings_arr.shape[1] != len(circ_layout):
            return {}
        logical_strings_arr = get_layout_state(strings_arr, circ_layout)
        fmt_str = "%i"*logical_strings_arr.shape[1]
        for i, state_string in enumerate(samples_dict.keys()):
            recovered_string = fmt_str % tuple(logical_strings_arr[i])
            new_layout_dict[recovered_string] = samples_dict[state_string]
        return new_layout_dict
    else:
        return {}

def measure_diagonal_observables(samples_dict, diagonal_observables, circ_layout=None, confidence=None):
    if len(samples_dict) == 0:
        if confidence is not None:
            return np.full(len(diagonal_observables), np.nan), np.full((len(diagonal_observables), 2), np.nan)
        else: 
            return np.full(len(diagonal_observables), np.nan)
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
    if confidence is not None:
        confidence_intervals = bootstrap_sample_dict_mean(samples_dict, strings_expectation_values, confidence)
        return expectation_values, confidence_intervals
    else:
        return expectation_values

def simulate_postselected_operators(fake_backend, sampler_opt_dict, transpiled_circuits, postselect_generating_func, observable_generating_funcs, return_samples_dicts=False, return_postselected_samples_dicts=False, odr_transpiled_circuits=None):
    nqubits = np.max([count_non_idle_qubits(circ) for circ in transpiled_circuits])
    postselection_ops, diagonal_observables, non_diagonal_observables, basis = initialize_postselection(nqubits, postselect_generating_func, observable_generating_funcs)
    # Express diagonal observables in the computational basis
    if basis != "Z":
        basis_changed_circuits = [append_basis_change_circuit(circ, basis, fake_backend) for circ in transpiled_circuits]
    else:
        basis_changed_circuits = transpiled_circuits
    observable_matrix = np.zeros((len(basis_changed_circuits), len(diagonal_observables + non_diagonal_observables)))
    if odr_transpiled_circuits is not None:
        if len(transpiled_circuits) != len(odr_transpiled_circuits):
            raise ValueError("One ODR circuit must be provided for each transpiled_circuit")
        p_matrix = np.zeros_like(observable_matrix)
        clean_observable_matrix = np.zeros_like(p_matrix)
        circuits_to_run = [basis_changed_circuits[i//2] if i % 2 == 0 else odr_transpiled_circuits[i//2] for i in range(2*len(basis_changed_circuits))]
    else:
        circuits_to_run = basis_changed_circuits
    # Execute diagonal circuits
    jobs = execute_sampler_batch(fake_backend, sampler_opt_dict, basis_changed_circuits)
    # Execute non diagonal circuits
    non_diagonal_obs_R_invs = []
    non_diagonal_obs_phases = []
    affected_postselection_operators = np.zeros((len(non_diagonal_observables), len(postselection_ops)), dtype=bool)
    for i, observable in enumerate(non_diagonal_observables):
        raise NotImplementedError("Still to be tested and incompatible with ODR and Sample Recovery")
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
        postselected_samples_dict = get_recovered_postselected_samples_dict(samples_dict, postselection_ops, circ_layout=basis_changed_circuits[i].layout.final_index_layout())
        if return_postselected_samples_dicts: postselected_samples_dicts.append(postselected_samples_dict)
        expectation_values = measure_diagonal_observables(samples_dict, diagonal_observables, circ_layout=basis_changed_circuits[i].layout.final_index_layout())
        observable_matrix[i, :len(diagonal_observables)] = expectation_values
    # TODO: Measure non-diagonal observables
    for i, observable in enumerate(non_diagonal_observables):
        this_observable_jobs = jobs[(i+1)*len(basis_changed_circuits):(i+2)*len(basis_changed_circuits)]
        this_postselection_operators = None
    to_return = [observable_matrix]
    if return_samples_dicts:
        to_return.append(samples_dicts)
    if return_postselected_samples_dicts:
        to_return.append(postselected_samples_dicts)
    return to_return if len(to_return) > 1 else to_return[0]

def load_job_from_local_or_ibmq(job_id, ibmq_service, jobs_result_folder):
    if os.path.isdir(jobs_result_folder):
        job_filepath = os.path.join(jobs_result_folder, f"{job_id}.json")
        if os.path.isfile(job_filepath):
            job_result = load_job_result(job_filepath)
        else:
            job_obj = ibmq_service.job(job_id=job_id)
            try:
                job_result = save_job_result(job_filepath, job_obj)
            except (RuntimeJobFailureError, RuntimeInvalidStateError):
                job_result = None
    else:
        job_obj = ibmq_service.job(job_id=job_id)
        try:
            job_result = job_obj.result()
        except RuntimeJobFailureError:
            job_result = None
    return job_result

def multiprocess_nrepetitions_combine_samples_dicts(samples_dicts, transpiled_circuits_shape, nrepetitions):
    indices = np.array(list(product(range(transpiled_circuits_shape[0]), range(transpiled_circuits_shape[1]))))
    grouped_samples_dicts = []
    for i_layout in range(transpiled_circuits_shape[0]):
        this_layout_samples = []
        for i_circ in range(transpiled_circuits_shape[1]):
            this_layout_samples.append(samples_dicts[i_layout, nrepetitions*i_circ:nrepetitions*(i_circ+1)])
        grouped_samples_dicts.append(tuple(this_layout_samples))
    grouped_samples_dicts = np.array(grouped_samples_dicts, dtype=object)
    grouped_samples_dicts = grouped_samples_dicts.reshape((-1, grouped_samples_dicts.shape[-1]))
    with mp.Pool(processes=mp.cpu_count()) as pool:
        combined_samples_dicts = pool.starmap(combine_samples_dicts, grouped_samples_dicts)
    combined_samples_dicts = np.array(combined_samples_dicts).reshape(transpiled_circuits_shape)
    return combined_samples_dicts.reshape(transpiled_circuits_shape)

def multiprocess_get_logical_samples_dicts(samples_dicts, layouts, recovery_function=None):
    samples_dicts = np.array(samples_dicts)
    layouts = np.array(layouts)
    if samples_dicts.shape != layouts.shape[:-1]:
        raise ValueError("samples_dicts and layouts do not have the appropriate shape")
    indices = np.array(list(product(range(samples_dicts.shape[0]), range(samples_dicts.shape[1]))))
    with mp.Pool(processes=mp.cpu_count()) as pool:
        logical_samples_dicts = pool.starmap(get_logical_samples_dict, [(samples_dicts[ind[0], ind[1]], layouts[ind[0], ind[1]][::-1]) for ind in indices])
    return np.array(logical_samples_dicts).reshape(samples_dicts.shape)

def multiprocess_layouts_combine_samples_dicts(samples_dicts):
    samples_dicts = np.array(samples_dicts)
    icirc_indices = np.arange(samples_dicts.shape[1])
    with mp.Pool(processes=mp.cpu_count()) as pool:
        combined_samples_dicts = pool.starmap(combine_samples_dicts, [tuple(samples_dicts[:, i]) for i in icirc_indices])
    return np.array(combined_samples_dicts)

def get_recovered_dict_with_basis(samples_dict, postselection_ops, basis, flip_threshold=None, minimum_shots=None, flipped_syndromes=None):
    if len(samples_dict) == 0:
        return {}
    strings_arr = np.array([[int(c) for c in string] for string in samples_dict.keys()])
    postselection_mask = np.array([[int(str(opel) == basis) for opel in operator] for operator in postselection_ops])
    if flipped_syndromes is not None:
        if len(flipped_syndromes) != postselection_mask.shape[0]:
            raise ValueError("flipped syndromes must be the same length as postselection_ops")
    decoder = Matching(postselection_mask)
    string_syndromes = (strings_arr @ postselection_mask.T) % 2
    if flipped_syndromes is not None:
        string_syndromes = (string_syndromes + flipped_syndromes) % 2
    predicted_flips = decoder.decode_batch(string_syndromes)
    # predicted_flips = decode_with_optimized_weights(string_syndromes, postselection_mask)
    npredicted_flips = predicted_flips.sum(axis=1)
    recovered_strings = (strings_arr + predicted_flips) % 2
    nflip_sort_mask = np.argsort(npredicted_flips)
    npredicted_flips = npredicted_flips[nflip_sort_mask]
    sampled_states = np.array(list(samples_dict.keys()))[nflip_sort_mask]
    recovered_strings = recovered_strings[nflip_sort_mask]
    recovered_samples_dict = {}
    fmt_str = "%i"*recovered_strings.shape[1]
    added_shots = 0
    for i, string in enumerate(sampled_states):
        recovered_string = fmt_str % tuple(recovered_strings[i])
        shots_to_add = samples_dict[string]
        if recovered_samples_dict.get(recovered_string, None) is not None:
            recovered_samples_dict[recovered_string] = recovered_samples_dict[recovered_string] + shots_to_add
        else:
            recovered_samples_dict[recovered_string] = samples_dict[string]
        added_shots += shots_to_add
        if (flip_threshold is not None) and (npredicted_flips[(i + 1) % len(npredicted_flips)] > flip_threshold):
            if minimum_shots is not None:
                if added_shots > minimum_shots:
                    break
            else:
                break
            
    return recovered_samples_dict

def multiprocess_get_postselected_recovered_samples_dicts(samples_dicts, postselection_ops, flip_threshold=None, minimum_shots=None, flipped_syndromes=None, recovery_function=None):
    all_diagonal, basis = diagonal_operators_check(postselection_ops)
    if recovery_function is None:
        recovery_function = get_recovered_dict_with_basis
    if not all_diagonal:
        raise ValueError("Only supports diagonal postselection operators in some basis")
    try:
        postselection_mask = np.array([[int(str(opel) == basis) for opel in operator] for operator in postselection_ops])
    except ValueError:
        raise ValueError("All postselection operators and samples must have the same number of qubits")
    samples_dicts = np.array(samples_dicts)
    og_samples_dicts_shape = samples_dicts.shape
    samples_dicts = samples_dicts.flatten()
    indices = np.arange(len(samples_dicts))
    with mp.Pool(processes=mp.cpu_count()) as pool:
        recovered_samples_dicts = pool.starmap(recovery_function, [(samples_dicts[i], postselection_ops, basis, flip_threshold, minimum_shots, flipped_syndromes) for i in indices])
    return np.array(recovered_samples_dicts).reshape(og_samples_dicts_shape)

def measure_diagonal_observables_with_basis(samples_dict, diagonal_observables, basis, confidence=None, shots=None):
    if len(samples_dict) == 0:
        if confidence is not None:
            return np.full(len(diagonal_observables), np.nan), np.full((len(diagonal_observables), 2), np.nan)
        else: 
            return np.full(len(diagonal_observables), np.nan)
    strings_expectation_values = np.zeros((len(samples_dict), len(diagonal_observables)))
    observables_mask = np.array([[int(str(opel) == basis) for opel in observable] for observable in diagonal_observables])
    for i, state_string in enumerate(samples_dict.keys()):
        state_string_rep = np.repeat(np.array([[int(c) for c in state_string]]), [len(diagonal_observables)], axis=0)
        observables_expectation_value = (-1)**(np.sum(state_string_rep*observables_mask, axis=1) % 2)
        strings_expectation_values[i] = observables_expectation_value
    weights = np.array(list(samples_dict.values()))/np.sum(list(samples_dict.values()))
    strings_expectation_times_weights = strings_expectation_values * weights.reshape((len(samples_dict), 1))
    expectation_values = np.sum(strings_expectation_times_weights, axis=0)
    if confidence is not None:
        if list(samples_dict.values())[0] < 1:
            if shots is None:
                raise ValueError("To boostrap a probability dictionary, shots must be provided")
            bootstrap_samples = {k:int(np.floor(v*shots)) for k, v in samples_dict.items()}
        else:
            bootstrap_samples = samples_dict
        confidence_intervals = bootstrap_sample_dict_mean(bootstrap_samples, strings_expectation_values, confidence)
        return expectation_values, confidence_intervals
    else:
        return expectation_values

def multiprocess_measure_diagonal_observables(samples_dicts, diagonal_observables, confidence=None, shots=None):
    nqubits = len(diagonal_observables[0])
    for observable in diagonal_observables:
        if len(observable) != nqubits:
            raise ValueError("All observables must act on the same number of qubits")
    for state_string in samples_dicts.flatten()[0].keys():
        if len(state_string) != nqubits:
            raise ValueError("The state strings and all observables must contain the same number of qubits")
    all_diagonal, basis = diagonal_operators_check(diagonal_observables)
    if not all_diagonal:
        raise ValueError("The provided observables are not diagonal in the same basis")
    samples_dicts = np.array(samples_dicts)
    og_samples_dicts_shape = samples_dicts.shape
    samples_dicts = samples_dicts.flatten()
    indices = np.arange(len(samples_dicts))
    if (confidence is not None) and (shots is not None):
        shots = np.array(shots)
        if shots.shape != og_samples_dicts_shape:
            raise ValueError("If provided, shots must have the same shape as samples_dicts")
        shots = shots.flatten()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        result = pool.starmap(measure_diagonal_observables_with_basis, [(samples_dicts[i], diagonal_observables, basis, confidence, shots[i] if shots is not None else None) for i in indices])
    if confidence is not None:
        expectation_values, confidence_intervals = np.array([result[i][0] for i in indices]), np.array([result[i][1] for i in indices])
        expectation_values = expectation_values.reshape((*og_samples_dicts_shape, len(diagonal_observables)))
        confidence_intervals = confidence_intervals.reshape((*og_samples_dicts_shape, len(diagonal_observables), 2))
        return expectation_values, confidence_intervals
    else:
        expectation_values = np.array(result).reshape((*og_samples_dicts_shape, len(diagonal_observables)))
        return expectation_values

def many_layouts_count_nqubits(circuits):
    if len(circuits.shape) > 1:
        nqubits = 0
        for circ in circuits.flatten():
            if (this_nqb := count_non_idle_qubits(circ)) > nqubits:
                nqubits = this_nqb
    else:
        nqubits = np.max([count_non_idle_qubits(circ) for circ in circuits])
    return nqubits

def load_samples_jobs(job_db, ibmq_service, sampler_opt_dict, transpiled_circuits, postselect_generating_func=None, extra_options=None, job_index=0, recovery=True, flip_threshold=None, minimum_shots=None, jobs_result_folder="", return_layouts=None, cal_circuits=None, postselect_cal=False, max_shots=75000, diagonal_basis="Z", verbose=False):
    # Preparation    
    transpiled_circuits = numpy_array_of_circuits(transpiled_circuits)
    if len(transpiled_circuits.shape) < 2:
        transpiled_circuits = transpiled_circuits[None]
    if cal_circuits is not None: 
        cal_circuits = numpy_array_of_circuits(cal_circuits)
        if len(cal_circuits.shape) < 2:
            cal_circuits = cal_circuits[None]
    nqubits = many_layouts_count_nqubits(transpiled_circuits)
    if postselect_generating_func is not None:
        postselection_ops = PauliList(set(postselect_generating_func(nqubits)))
        post_selection_diagonal, diagonal_basis = diagonal_operators_check(postselection_ops)
        if not post_selection_diagonal:
            raise ValueError("Only supports diagonal postselection operators in some basis")
    extra_options = {} if extra_options is None else extra_options
    options = extra_options | sampler_opt_dict | ({"cal": True} if cal_circuits is not None else {})
    if cal_circuits is not None:
        if transpiled_circuits.shape != cal_circuits.shape:
            raise ValueError("One calibration circuit must be provided for each transpiled_circuit")
    
    if len(transpiled_circuits.shape) > 1:
        shots = sampler_opt_dict.get("default_shots", 1)
        shots_per_layout = shots // transpiled_circuits.shape[0]
        if shots_per_layout == 0:
            shots_per_layout = 1
    else:
        shots_per_layout = sampler_opt_dict.get("default_shots", 1)

    if shots_per_layout > max_shots:
        nrepetitions = shots_per_layout // max_shots
        if nrepetitions == 0:
            nrepetitions = 1
    else:
        nrepetitions = 1

    ran_circuits = generate_multirep_circuits_to_run(transpiled_circuits, cal_circuits, nrepetitions)
    
    # Load jobs from local or IBMQ
    if verbose: print("\rLoading jobs...".ljust(100), end="")
    jobs_layout = []
    jobs_db_json = job_db.search_by_params(options, ran_circuits.flatten().tolist(), "Sampler", strict_depth=False, limit=job_index+1)
    if job_index != 0:
        jobs_db_json = jobs_db_json[job_index]
    if not jobs_db_json:
        raise ValueError("No jobs found with the provided settings")
    job_ids = [jobs_json["job_id"] for jobs_json in jobs_db_json["jobs"]]
    jobs_layout = [job_json["layout"] for job_json in jobs_db_json["jobs"]]

    jobs = []
    for job_id in job_ids:
        if verbose: print(f"\rLoading job {job_id}...".ljust(100), end="")
        job = load_job_from_local_or_ibmq(job_id, ibmq_service, jobs_result_folder)
        jobs.append(job)
    if cal_circuits is not None:
        newdims = (transpiled_circuits.shape[0], cal_circuits.shape[1] + nrepetitions*transpiled_circuits.shape[1])
    else:
        newdims = (transpiled_circuits.shape[0], nrepetitions*transpiled_circuits.shape[1])
    jobs = np.array(jobs, dtype=object).reshape(newdims)
    jobs_layout = np.array(jobs_layout).reshape((*newdims, nqubits))

    if cal_circuits is not None:
        cal_mask = np.tile([True] + [False]*nrepetitions, transpiled_circuits.shape[1])
        cal_jobs = jobs[:, cal_mask]
        real_jobs = jobs[:, ~cal_mask]
    else:
        real_jobs = jobs

    # Load samples dicts from jobs and organize them in memory
    if cal_circuits is not None:
        cal_samples_dicts = []
        diagonal_cal_jobs = cal_jobs[:, :cal_circuits.shape[-1]]
        for this_layout_cal_jobs in diagonal_cal_jobs:
            this_layout_samples_dicts = []
            for dcaljob in this_layout_cal_jobs:
                if dcaljob is not None:
                    try:
                        dcaljob = dcaljob[0]
                    except IndexError:
                        pass
                    except TypeError:
                        pass
                    cal_samples_dict = list(dict(dcaljob.data).values())[0].get_counts()
                else:
                    cal_samples_dict = {}
                this_layout_samples_dicts.append(cal_samples_dict)
            cal_samples_dicts.append(this_layout_samples_dicts)
        cal_samples_dicts = np.array(cal_samples_dicts)

    real_samples_dicts = []
    diagonal_real_jobs = real_jobs[:, :nrepetitions*transpiled_circuits.shape[1]]
    for this_layout_real_jobs in diagonal_real_jobs:
        this_layout_samples_dicts = []
        for i, drealjob in enumerate(this_layout_real_jobs):
            if drealjob is not None:
                try:
                    drealjob = drealjob[0]
                except IndexError:
                    pass
                except TypeError:
                    pass
                real_samples_dict = list(dict(drealjob.data).values())[0].get_counts()
            else:
                real_samples_dict = {}
            this_layout_samples_dicts.append(real_samples_dict)
        real_samples_dicts.append(this_layout_samples_dicts)
    real_samples_dicts = np.array(real_samples_dicts)

    to_return = []
    if postselect_generating_func is not None:
        # Generate postselected samples dicts
        if verbose: print("\rGenerating postselected samples...".ljust(100), end="")
        if recovery is None:
            pass
        elif not recovery:
            flip_threshold = 0
            recovery = None
        elif recovery == True:
            recovery = None
        postselected_real_samples_dicts = multiprocess_get_postselected_recovered_samples_dicts(real_samples_dicts, postselection_ops, flip_threshold, minimum_shots, recovery_function=recovery)
        if cal_circuits is not None:
            if postselect_cal:
                postselected_odr_samples_dicts = multiprocess_get_postselected_recovered_samples_dicts(cal_samples_dicts, postselection_ops, flip_threshold, minimum_shots, recovery_function=recovery)
            else:
                postselected_odr_samples_dicts = cal_samples_dicts
        to_return.append((postselected_real_samples_dicts, postselected_odr_samples_dicts))
    to_return.append((real_samples_dicts, cal_samples_dicts))

    if return_layouts:
        to_return.append(jobs_layout)
    
    return to_return if len(to_return) > 1 else to_return[0]

def load_postselected_jobs_obs_samples(samples_dicts, postselect_generating_func, observable_generating_funcs, recovery=True, flip_threshold=None, minimum_shots=None, confidence=None, return_samples_dicts=False, return_postselected_samples_dicts=False, postselect_odr=True, odr_expected_observable=1, max_shots=100000, verbose=False):
    odr = len(samples_dicts) == 2
    if odr:
        real_samples_dicts, odr_samples_dicts = samples_dicts
    else:
        real_samples_dicts = samples_dicts

    nqubits = len(list(real_samples_dicts[..., 0][0].keys())[0])
    postselection_ops, diagonal_observables, non_diagonal_observables, basis = initialize_postselection(nqubits, postselect_generating_func, observable_generating_funcs)

    # if len(real_samples_dicts.shape) > 1:
    #     shots = sum(real_samples_dicts[..., 0][0].values())
    #     shots_per_layout = shots // real_samples_dicts.shape[0]
    #     if shots_per_layout == 0:
    #         shots_per_layout = 1
    # else:
    #     shots_per_layout = sum(real_samples_dicts[..., 0].values())
    #     transpiled_circuits = transpiled_circuits[None]

    # if shots_per_layout > max_shots:
    #     nrepetitions = shots_per_layout // max_shots
    #     if nrepetitions == 0:
    #         nrepetitions = 1
    # else:
    #     nrepetitions = 1

    # Generate postselected samples dicts
    if verbose: print("\rGenerating postselected samples...".ljust(100), end="")
    if recovery is None:
        pass
    elif not recovery:
        flip_threshold = 0
        recovery = None
    elif recovery == True:
        recovery = None
    postselected_real_samples_dicts = multiprocess_get_postselected_recovered_samples_dicts(real_samples_dicts, postselection_ops, flip_threshold, minimum_shots, recovery_function=recovery)
    if odr is not None:
        if postselect_odr:
            postselected_odr_samples_dicts = multiprocess_get_postselected_recovered_samples_dicts(odr_samples_dicts, postselection_ops, flip_threshold, minimum_shots, recovery_function=recovery)
        else:
            postselected_odr_samples_dicts = odr_samples_dicts

    # Combine samples dicts from different repetitions
    # if verbose: print("\rCombining samples dicts...".ljust(100), end="")
    # if nrepetitions > 1:
    #     real_samples_dicts = multiprocess_nrepetitions_combine_samples_dicts(real_samples_dicts, (real_samples_dicts.shape[0], real_samples_dicts.shape[1]//nrepetitions), nrepetitions)
    #     postselected_real_samples_dicts = multiprocess_nrepetitions_combine_samples_dicts(postselected_real_samples_dicts, (postselected_real_samples_dicts.shape[0], postselected_real_samples_dicts.shape[1]//nrepetitions), nrepetitions)
    #     # if odr_circuits is not None:
    #     #     odr_samples_dicts = multiprocess_nrepetitions_combine_samples_dicts(odr_samples_dicts, transpiled_circuits.shape, nrepetitions)
    #     #     postselected_odr_samples_dicts = multiprocess_nrepetitions_combine_samples_dicts(postselected_odr_samples_dicts, transpiled_circuits.shape, nrepetitions)

    # Combine samples dicts from different layouts
    combined_real_samples_dicts = multiprocess_layouts_combine_samples_dicts(real_samples_dicts)
    combined_postselected_real_samples_dicts = multiprocess_layouts_combine_samples_dicts(postselected_real_samples_dicts)
    if odr is not None:
        combined_postselected_odr_samples_dicts = multiprocess_layouts_combine_samples_dicts(postselected_odr_samples_dicts)

    # Measure diagonal observables and errors from samples
    if verbose: print("\rMeasuring observables...".ljust(100), end="")
    real_expectation_values, real_confidence_intervals = multiprocess_measure_diagonal_observables(combined_postselected_real_samples_dicts, diagonal_observables, 0 if confidence is None else confidence)
    raw_expectation_values, raw_confidence_intervals = multiprocess_measure_diagonal_observables(combined_real_samples_dicts, diagonal_observables, 0 if confidence is None else confidence)
    if odr is not None:
        odr_expectation_values, odr_confidence_intervals = multiprocess_measure_diagonal_observables(combined_postselected_odr_samples_dicts, diagonal_observables, 0 if confidence is None else confidence)
        clean_expectation_values = np.zeros_like(real_expectation_values)
        clean_confidence_intervals = np.zeros_like(real_confidence_intervals)
        for i in range(len(real_expectation_values)):
            p_arr = compute_depolarizing_probability(odr_expectation_values[i], odr_expected_observable, 0, nqubits)
            clean_expectation_values[i], clean_confidence_intervals[i] = renormalize_observable(real_expectation_values[i], p_arr, 0, nqubits, observable_range=None, expected_observable_arr=odr_expected_observable, observable_confidence=real_confidence_intervals[i])

    # Measure non-diagonal observables
    for observable in non_diagonal_observables:
        raise NotImplementedError("Non-diagonal observables not yet supported")

    # Return
    if verbose: print(f"\rLoaded {len(postselected_real_samples_dicts.flatten())} jobs".ljust(100))
    if odr is None:
        to_return = [[real_expectation_values, raw_expectation_values]]
        if confidence is not None:
            to_return.append([real_confidence_intervals, raw_confidence_intervals])
        if return_samples_dicts:
            to_return.append([real_samples_dicts])
        if return_postselected_samples_dicts:
            to_return.append([postselected_real_samples_dicts])
        return to_return if len(to_return) > 1 else to_return[0]
    else:
        to_return = [[clean_expectation_values, real_expectation_values, odr_expectation_values, raw_expectation_values]]
        if confidence is not None:
            to_return.append([clean_confidence_intervals, real_confidence_intervals, odr_confidence_intervals, raw_confidence_intervals])
        if return_samples_dicts:
            to_return.append([real_samples_dicts, odr_samples_dicts])
        if return_postselected_samples_dicts:
            to_return.append([postselected_real_samples_dicts, postselected_odr_samples_dicts])
        return to_return if len(to_return) > 1 else to_return[0]

def load_postselected_jobs_obs(job_db, ibmq_service, sampler_opt_dict, transpiled_circuits, postselect_generating_func, observable_generating_funcs, extra_options=None, job_index=0, recovery=True, flip_threshold=None, minimum_shots=None, jobs_result_folder="", confidence=None, return_samples_dicts=False, return_postselected_samples_dicts=False, return_layouts=False, odr_circuits=None, postselect_odr=True, odr_expected_observable=1, max_shots=75000, verbose=False):
    transpiled_circuits = numpy_array_of_circuits(transpiled_circuits)
    odr_circuits = numpy_array_of_circuits(odr_circuits)
    nqubits = many_layouts_count_nqubits(transpiled_circuits)

    postselection_ops, diagonal_observables, non_diagonal_observables, basis = initialize_postselection(nqubits, postselect_generating_func, observable_generating_funcs)
    
    (postselected_real_samples_dicts, postselected_odr_samples_dicts), (real_samples_dicts, odr_samples_dicts), jobs_layout = load_samples_jobs(job_db, ibmq_service, sampler_opt_dict, transpiled_circuits, postselect_generating_func, extra_options, job_index, recovery, flip_threshold, minimum_shots, jobs_result_folder, return_layouts=True, cal_circuits=odr_circuits, postselect_cal=postselect_odr, max_shots=max_shots, verbose=verbose)

    if len(transpiled_circuits.shape) > 1:
        shots = sampler_opt_dict.get("default_shots", 1)
        shots_per_layout = shots // transpiled_circuits.shape[0]
        if shots_per_layout == 0:
            shots_per_layout = 1
    else:
        shots_per_layout = sampler_opt_dict.get("default_shots", 1)
        transpiled_circuits = transpiled_circuits[None]

    if shots_per_layout > max_shots:
        nrepetitions = shots_per_layout // max_shots
        if nrepetitions == 0:
            nrepetitions = 1
    else:
        nrepetitions = 1

    # Combine samples dicts from different repetitions
    if verbose: print("\rCombining samples dicts...".ljust(100), end="")
    if nrepetitions > 1:
        real_samples_dicts = multiprocess_nrepetitions_combine_samples_dicts(real_samples_dicts, transpiled_circuits.shape, nrepetitions)
        postselected_real_samples_dicts = multiprocess_nrepetitions_combine_samples_dicts(postselected_real_samples_dicts, transpiled_circuits.shape, nrepetitions)
        # if odr_circuits is not None:
        #     odr_samples_dicts = multiprocess_nrepetitions_combine_samples_dicts(odr_samples_dicts, transpiled_circuits.shape, nrepetitions)
        #     postselected_odr_samples_dicts = multiprocess_nrepetitions_combine_samples_dicts(postselected_odr_samples_dicts, transpiled_circuits.shape, nrepetitions)

    # Combine samples dicts from different layouts
    combined_real_samples_dicts = multiprocess_layouts_combine_samples_dicts(real_samples_dicts)
    combined_postselected_real_samples_dicts = multiprocess_layouts_combine_samples_dicts(postselected_real_samples_dicts)
    if odr_circuits is not None:
        combined_postselected_odr_samples_dicts = multiprocess_layouts_combine_samples_dicts(postselected_odr_samples_dicts)

    # Measure diagonal observables and errors from samples
    if verbose: print("\rMeasuring observables...".ljust(100), end="")
    real_expectation_values, real_confidence_intervals = multiprocess_measure_diagonal_observables(combined_postselected_real_samples_dicts, diagonal_observables, 0 if confidence is None else confidence)
    raw_expectation_values, raw_confidence_intervals = multiprocess_measure_diagonal_observables(combined_real_samples_dicts, diagonal_observables, 0 if confidence is None else confidence)
    if odr_circuits is not None:
        odr_expectation_values, odr_confidence_intervals = multiprocess_measure_diagonal_observables(combined_postselected_odr_samples_dicts, diagonal_observables, 0 if confidence is None else confidence)
        clean_expectation_values = np.zeros_like(real_expectation_values)
        clean_confidence_intervals = np.zeros_like(real_confidence_intervals)
        for i in range(len(real_expectation_values)):
            p_arr = compute_depolarizing_probability(odr_expectation_values[i], odr_expected_observable, 0, odr_circuits.flatten()[i].num_qubits)
            clean_expectation_values[i], clean_confidence_intervals[i] = renormalize_observable(real_expectation_values[i], p_arr, 0, transpiled_circuits.flatten()[i].num_qubits, observable_range=(-1, 1), expected_observable_arr=1, observable_confidence=real_confidence_intervals[i])

    # Measure non-diagonal observables
    for observable in non_diagonal_observables:
        raise NotImplementedError("Non-diagonal observables not yet supported")

    # Return
    if verbose: print(f"\rLoaded {len(postselected_real_samples_dicts.flatten())} jobs".ljust(100))
    if odr_circuits is None:
        to_return = [[real_expectation_values, raw_expectation_values]]
        if confidence is not None:
            to_return.append([real_confidence_intervals, raw_confidence_intervals])
        if return_samples_dicts:
            to_return.append([real_samples_dicts])
        if return_postselected_samples_dicts:
            to_return.append([postselected_real_samples_dicts])
        if return_layouts:
            to_return.append(jobs_layout)
        return to_return if len(to_return) > 1 else to_return[0]
    else:
        to_return = [[clean_expectation_values, real_expectation_values, odr_expectation_values, raw_expectation_values]]
        if confidence is not None:
            to_return.append([clean_confidence_intervals, real_confidence_intervals, odr_confidence_intervals, raw_confidence_intervals])
        if return_samples_dicts:
            to_return.append([real_samples_dicts, odr_samples_dicts])
        if return_postselected_samples_dicts:
            to_return.append([postselected_real_samples_dicts, postselected_odr_samples_dicts])
        if return_layouts:
            to_return.append(jobs_layout)
        return to_return if len(to_return) > 1 else to_return[0]
    
def load_postselected_jobs_inverse_order(job_db, ibmq_service, sampler_opt_dict, transpiled_circuits, postselect_generating_func, observable_generating_funcs, extra_options=None, job_index=0, recovery=True, flip_threshold=None, minimum_shots=None, jobs_result_folder="", confidence=None, return_samples_dicts=False, return_postselected_samples_dicts=False, return_layouts=False, odr_circuits=None, postselect_odr=True, max_shots=75000, verbose=False):
    transpiled_circuits = numpy_array_of_circuits(transpiled_circuits)
    odr_circuits = numpy_array_of_circuits(odr_circuits)
    nqubits = many_layouts_count_nqubits(transpiled_circuits)

    postselection_ops, diagonal_observables, non_diagonal_observables, basis = initialize_postselection(nqubits, postselect_generating_func, observable_generating_funcs)
    
    (postselected_real_samples_dicts, postselected_odr_samples_dicts), (real_samples_dicts, odr_samples_dicts), jobs_layout = load_samples_jobs(job_db, ibmq_service, sampler_opt_dict, transpiled_circuits, postselect_generating_func, extra_options, job_index, recovery, flip_threshold, minimum_shots, jobs_result_folder, return_layouts=True, cal_circuits=odr_circuits, postselect_cal=postselect_odr, max_shots=max_shots, verbose=verbose)

    if len(transpiled_circuits.shape) > 1:
        shots = sampler_opt_dict.get("default_shots", 1)
        shots_per_layout = shots // transpiled_circuits.shape[0]
        if shots_per_layout == 0:
            shots_per_layout = 1
    else:
        shots_per_layout = sampler_opt_dict.get("default_shots", 1)
        transpiled_circuits = transpiled_circuits[None]

    if shots_per_layout > max_shots:
        nrepetitions = shots_per_layout // max_shots
        if nrepetitions == 0:
            nrepetitions = 1
    else:
        nrepetitions = 1

    # Combine samples dicts from different repetitions
    if verbose: print("\rCombining samples dicts...".ljust(100), end="")
    if nrepetitions > 1:
        real_samples_dicts = multiprocess_nrepetitions_combine_samples_dicts(real_samples_dicts, transpiled_circuits.shape, nrepetitions)
        postselected_real_samples_dicts = multiprocess_nrepetitions_combine_samples_dicts(postselected_real_samples_dicts, transpiled_circuits.shape, nrepetitions)
        # if odr_circuits is not None:
        #     odr_samples_dicts = multiprocess_nrepetitions_combine_samples_dicts(odr_samples_dicts, transpiled_circuits.shape, nrepetitions)
        #     postselected_odr_samples_dicts = multiprocess_nrepetitions_combine_samples_dicts(postselected_odr_samples_dicts, transpiled_circuits.shape, nrepetitions)

    # Combine samples dicts from different layouts
    combined_real_samples_dicts = multiprocess_layouts_combine_samples_dicts(real_samples_dicts)
    combined_postselected_real_samples_dicts = multiprocess_layouts_combine_samples_dicts(postselected_real_samples_dicts)
    if odr_circuits is not None:
        combined_odr_samples_dicts = multiprocess_layouts_combine_samples_dicts(odr_samples_dicts)
        combined_postselected_odr_samples_dicts = multiprocess_layouts_combine_samples_dicts(postselected_odr_samples_dicts)
    
    # Measure diagonal observables and errors from samples
    if verbose: print("\rMeasuring observables...".ljust(100), end="")
    real_expectation_values, real_confidence_intervals = multiprocess_measure_diagonal_observables(combined_postselected_real_samples_dicts, diagonal_observables, 0 if confidence is None else confidence)
    raw_expectation_values, raw_confidence_intervals = multiprocess_measure_diagonal_observables(combined_real_samples_dicts, diagonal_observables, 0 if confidence is None else confidence)
    if odr_circuits is not None:
        odr_expectation_values, odr_confidence_intervals = multiprocess_measure_diagonal_observables(combined_odr_samples_dicts, diagonal_observables, 0 if confidence is None else confidence)
        clean_expectation_values = np.zeros_like(real_expectation_values)
        clean_confidence_intervals = np.zeros_like(real_confidence_intervals)
        for i in range(len(real_expectation_values)):
            p_arr = compute_depolarizing_probability(odr_expectation_values[i], 1, 0, odr_circuits.flatten()[i].num_qubits)
            clean_expectation_values[i], clean_confidence_intervals[i] = renormalize_observable(real_expectation_values[i], p_arr, 0, transpiled_circuits.flatten()[i].num_qubits, observable_range=(-1, 1), expected_observable_arr=1, observable_confidence=real_confidence_intervals[i])

    # Measure non-diagonal observables
    for observable in non_diagonal_observables:
        raise NotImplementedError("Non-diagonal observables not yet supported")

    # Return
    if verbose: print(f"\rLoaded {len(postselected_real_samples_dicts.flatten())} jobs".ljust(100))
    if odr_circuits is None:
        to_return = [[real_expectation_values, raw_expectation_values]]
        if confidence is not None:
            to_return.append([real_confidence_intervals, raw_confidence_intervals])
        if return_samples_dicts:
            to_return.append([real_samples_dicts])
        if return_postselected_samples_dicts:
            to_return.append([postselected_real_samples_dicts])
        if return_layouts:
            to_return.append(jobs_layout)
        return to_return if len(to_return) > 1 else to_return[0]
    else:
        to_return = [[clean_expectation_values, real_expectation_values, odr_expectation_values, raw_expectation_values]]
        if confidence is not None:
            to_return.append([clean_confidence_intervals, real_confidence_intervals, odr_confidence_intervals, raw_confidence_intervals])
        if return_samples_dicts:
            to_return.append([real_samples_dicts, odr_samples_dicts])
        if return_postselected_samples_dicts:
            to_return.append([postselected_real_samples_dicts, postselected_odr_samples_dicts])
        if return_layouts:
            to_return.append(jobs_layout)
        return to_return if len(to_return) > 1 else to_return[0]
    
def load_dec_jobs_obs(job_db, ibmq_service, sampler_opt_dict, transpiled_circuits, observable_generating_funcs, dec_circuits, postselect_generating_func=None, extra_options=None, job_index=0, recovery=True, flip_threshold=None, jobs_result_folder="", confidence=None, return_samples_dicts=False, return_postselected_samples_dicts=False, return_layouts=False, num_keep=2**20, max_shots=75000, diagonal_basis="Z", verbose=False):
    transpiled_circuits = numpy_array_of_circuits(transpiled_circuits)
    dec_circuits = numpy_array_of_circuits(dec_circuits)
    nqubits = many_layouts_count_nqubits(transpiled_circuits)

    if postselect_generating_func is not None:
        postselection_ops, diagonal_observables, non_diagonal_observables, basis = initialize_postselection(nqubits, postselect_generating_func, observable_generating_funcs)
        (real_samples_dicts, dec_samples_dicts), jobs_layout = load_samples_jobs(job_db, ibmq_service, sampler_opt_dict, transpiled_circuits, None, extra_options, job_index, recovery, flip_threshold, jobs_result_folder, return_layouts=True, cal_circuits=dec_circuits, postselect_cal=False, max_shots=max_shots, verbose=verbose)
    else:
        diagonal_observables, non_diagonal_observables = generate_observables(nqubits, observable_generating_funcs, diagonal_basis)
        (real_samples_dicts, dec_samples_dicts), jobs_layout = load_samples_jobs(job_db, ibmq_service, sampler_opt_dict, transpiled_circuits, None, extra_options, job_index, recovery, flip_threshold, jobs_result_folder, return_layouts=True, cal_circuits=dec_circuits, postselect_cal=False, max_shots=max_shots, verbose=verbose)

    if len(transpiled_circuits.shape) > 1:
        shots = sampler_opt_dict.get("default_shots", 1)
        shots_per_layout = shots // transpiled_circuits.shape[0]
        if shots_per_layout == 0:
            shots_per_layout = 1
    else:
        shots_per_layout = sampler_opt_dict.get("default_shots", 1)
        transpiled_circuits = transpiled_circuits[None]

    if shots_per_layout > max_shots:
        nrepetitions = shots_per_layout // max_shots
        if nrepetitions == 0:
            nrepetitions = 1
    else:
        nrepetitions = 1

    # Use DEC to mitigate errors
    if verbose: print("\rPerforming DEC mitigation...".ljust(100), end="")
    corrected_samples_dicts = dec.mitigate(real_samples_dicts.flatten(), dec_samples_dicts.flatten().tolist()*nrepetitions, [{"0"*nqubits:sum(sdict.values())} for sdict in real_samples_dicts.flatten()], num_keep=num_keep)
    # corrected_samples_dicts = dec.multiprocess_mitigate(real_samples_dicts.flatten(), dec_samples_dicts.flatten().tolist()*nrepetitions, [{"0"*nqubits:sum(sdict.values())} for sdict in real_samples_dicts.flatten()]) 
    corrected_samples_dicts = corrected_samples_dicts.reshape(real_samples_dicts.shape)
    # if corrected_samples_dicts.ndim == 1:
    #     corrected_samples_dicts = corrected_samples_dicts[None]

    # Combine samples dicts from different repetitions
    if verbose: print("\rCombining samples dicts...".ljust(100), end="")
    if nrepetitions > 1:
        real_samples_dicts = multiprocess_nrepetitions_combine_samples_dicts(real_samples_dicts, transpiled_circuits.shape, nrepetitions)
        dec_samples_dicts = multiprocess_nrepetitions_combine_samples_dicts(dec_samples_dicts, transpiled_circuits.shape, nrepetitions)
        corrected_samples_dicts = multiprocess_nrepetitions_combine_samples_dicts(corrected_samples_dicts, transpiled_circuits.shape, nrepetitions)

    # Postselect corrected dicts
    if postselect_generating_func is not None:
        if verbose: print("\rGenerating postselected samples...".ljust(100), end="")
        postselected_real_samples_dicts = multiprocess_get_postselected_recovered_samples_dicts(real_samples_dicts, postselection_ops, flip_threshold, recovery_function=None)
        postselected_dec_samples_dicts = multiprocess_get_postselected_recovered_samples_dicts(dec_samples_dicts, postselection_ops, flip_threshold, recovery_function=None)
        postselected_corrected_samples_dicts = multiprocess_get_postselected_recovered_samples_dicts(corrected_samples_dicts, postselection_ops, flip_threshold, recovery_function=None)

    # Combine samples dicts from different layouts
    combined_real_samples_dicts = multiprocess_layouts_combine_samples_dicts(real_samples_dicts)
    combined_dec_samples_dicts = multiprocess_layouts_combine_samples_dicts(dec_samples_dicts)
    combined_corrected_samples_dicts = multiprocess_layouts_combine_samples_dicts(corrected_samples_dicts)
    if postselect_generating_func is not None:
        combined_postselected_real_samples_dicts = multiprocess_layouts_combine_samples_dicts(postselected_real_samples_dicts)
        combined_postselected_dec_samples_dicts = multiprocess_layouts_combine_samples_dicts(postselected_dec_samples_dicts)
        combined_postselected_corrected_samples_dicts = multiprocess_layouts_combine_samples_dicts(postselected_corrected_samples_dicts)
        combined_postselected_corrected_samples_dicts = np.array([dec.normalize_dist_dict(samples) for samples in combined_postselected_corrected_samples_dicts], dtype=object).reshape(combined_postselected_real_samples_dicts.shape)
    
    # Measure diagonal observables and errors from samples
    if verbose: print("\rMeasuring observables...".ljust(100), end="")
    if postselect_generating_func is not None:
        real_expectation_values, real_confidence_intervals = multiprocess_measure_diagonal_observables(combined_postselected_real_samples_dicts, diagonal_observables, 0 if confidence is None else confidence)
        dec_expectation_values, dec_confidence_intervals = multiprocess_measure_diagonal_observables(combined_postselected_dec_samples_dicts, diagonal_observables, 0 if confidence is None else confidence)
        shots = np.array([sum(d.values()) for d in combined_postselected_real_samples_dicts.flatten()]).reshape(combined_postselected_real_samples_dicts.shape)
        corrected_expectation_values, corrected_confidence_intervals = multiprocess_measure_diagonal_observables(combined_postselected_corrected_samples_dicts, diagonal_observables, 0 if confidence is None else confidence, shots)
    else:
        real_expectation_values, real_confidence_intervals = multiprocess_measure_diagonal_observables(combined_real_samples_dicts, diagonal_observables, 0 if confidence is None else confidence)
        dec_expectation_values, dec_confidence_intervals = multiprocess_measure_diagonal_observables(combined_dec_samples_dicts, diagonal_observables, 0 if confidence is None else confidence)
        shots = np.array([sum(d.values()) for d in combined_postselected_real_samples_dicts.flatten()]).reshape(combined_postselected_real_samples_dicts.shape)
        corrected_expectation_values, corrected_confidence_intervals = multiprocess_measure_diagonal_observables(combined_corrected_samples_dicts, diagonal_observables, 0 if confidence is None else confidence, shots)

    # Measure non-diagonal observables
    for observable in non_diagonal_observables:
        raise NotImplementedError("Non-diagonal observables not yet supported")

    # Return
    if verbose: print(f"\rLoaded {len(real_samples_dicts.flatten())} simulations".ljust(100))
    to_return = [[corrected_expectation_values, real_expectation_values, dec_expectation_values]]
    if confidence is not None:
        to_return.append([corrected_confidence_intervals, real_confidence_intervals, dec_confidence_intervals])
    if return_samples_dicts:
        to_return.append([combined_corrected_samples_dicts, combined_real_samples_dicts, combined_dec_samples_dicts])
    if return_postselected_samples_dicts and postselect_generating_func is not None:
        to_return.append([combined_postselected_corrected_samples_dicts, combined_postselected_real_samples_dicts, combined_postselected_dec_samples_dicts])
    if return_layouts:
        to_return.append(jobs_layout)
    return to_return if len(to_return) > 1 else to_return[0]