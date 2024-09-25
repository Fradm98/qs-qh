from utils.circs import simplify_logical_circuits
from qiskit.quantum_info import Pauli, PauliList
from qiskit import QuantumCircuit
from collections import Counter
import sympy as sym
import numpy as np

# -----------------------------------------
#           BASIS CHANGE CIRCUITS
# -----------------------------------------

# AUXILIARY FUNCTIONS

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
        ndiagonal = str(pauli).count("I") + str(pauli).count("Z")
        this_non_diagonal_qubits = len(pauli) - ndiagonal
        if this_non_diagonal_qubits > max_non_diagonal_qubits:
            max_non_diagonal_qubits = this_non_diagonal_qubits
    return max_non_diagonal_qubits

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
        return f"InstructionList<{self.nqubits}qb>[{gates}]"
    
    def __getitem__(self, i):
        return self.instructions[i]

    def binary_representation_conjugation(self, S):
        for instruction in self.instructions:
            S = instruction.binary_representation_conjugation(S)
        return S
    
    def to_qiskit_circuit(self):
        qc = QuantumCircuit(self.nqubits)
        for instruction in self.instructions:
            if type(instruction) == H:
                qc.h(instruction.qubits)
            elif type(instruction) == P:
                qc.s(instruction.qubits)
            else:
                qc.cz(instruction.ctrl, instruction.target)
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
    zero_diag_elements = np.where(np.equal(C_diag, 0))[0]
    row_swaps = {}
    for zero_diag_el_ind in zero_diag_elements:
        non_zero_diag_col = S[:, zero_diag_el_ind]
        first_one_row_ind = np.where(np.equal(non_zero_diag_col, 1))[0][0]
        row_swaps[zero_diag_el_ind] = first_one_row_ind
    instructions = InstructionList(nrows)
    hadamard_qubits = [row_swaps.get(i, i) for i in np.arange(nrows - Sx_rank)]
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
    return S5, R_inv, instructions

def paulis_diagonalization_circuit(observables, postselection_ops):
    # Quantum 5, 385 (2021)
    postselection_ops = PauliList([Pauli(postselection_op) for postselection_op in postselection_ops])
    observables = PauliList([Pauli(observable) for observable in observables])
    all_commute = check_postselection_observable_commutation(observables, postselection_ops)
    if not all_commute:
        raise ValueError("All observables and postselection operators must commute")
    nqubits = len(postselection_ops[0])
    commuting_set = postselection_ops + observables
    non_diagonal_qubits_num = count_non_diagonal_qubits(commuting_set)
    if non_diagonal_qubits_num == 0:
        return QuantumCircuit(nqubits)
    Sp = binary_pauli_representation(commuting_set)
    redS, R_inv, instructions = reduce_binary_string_representation(Sp)
    return instructions.to_qiskit_circuit(), R_inv