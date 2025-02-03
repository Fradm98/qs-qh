import scipy.sparse as sparse
import numpy as np

# ---------------------------------------------------------------------
# FUNCTIONS PROVIDING THE SPARSE REPRESENTATION OF A TENSOR PRODUCT OF
# PAULI MATRICES
# ---------------------------------------------------------------------

def sparse_non_diag_paulis_indices(n, N):
    """Returns a tuple (row_indices, col_indices) containing the row and col indices of the non_zero elements
       of the tensor product of a non diagonal pauli matrix (x, y) acting over a single qubit in a Hilbert
       space of N qubits"""
    if 0 <= n < N:
        block_length = 2**(N - n - 1)
        nblocks = 2**n
        ndiag_elements = block_length*nblocks
        k = np.arange(ndiag_elements, dtype=int)
        red_row_col_ind = (k % block_length) + 2*(k // block_length)*block_length
        upper_diag_row_indices = red_row_col_ind
        upper_diag_col_indices = block_length + red_row_col_ind
        row_indices = np.concatenate((upper_diag_row_indices, upper_diag_col_indices))
        col_indices = np.concatenate((upper_diag_col_indices, upper_diag_row_indices))
        return row_indices, col_indices
    else:
        raise ValueError("Index n must fulfill 0 <= n < N")

def sparse_pauli_x(n, N, row_indices_cache=None, col_indices_cache=None):
    """Returns a CSC sparse matrix representation of the pauli_x matrix acting over qubit n in a Hilbert space of N qubits
       0 <= n < N"""
    if 0 <= n < N:
        if (row_indices_cache is None) or (col_indices_cache is None):
            row_indices_cache, col_indices_cache = sparse_non_diag_paulis_indices(n, N)
        data = np.ones_like(row_indices_cache)
        result = sparse.csc_array((data, (row_indices_cache, col_indices_cache)), shape=(2**N, 2**N), dtype=complex)
        return result
    else:
        raise ValueError("Index n must fulfill 0 <= n < N")

def sparse_pauli_y(n, N, row_indices_cache=None, col_indices_cache=None):
    """Returns a CSC sparse matrix representation of the pauli_y matrix acting over qubit n in a Hilbert space of N qubits
       0 <= n < N"""
    if 0 <= n < N :
        if (row_indices_cache is None) or (col_indices_cache is None):
            row_indices_cache, col_indices_cache = sparse_non_diag_paulis_indices(n, N)
        data = -1j*np.ones_like(row_indices_cache)
        data[len(data)//2::] = 1j
        result = sparse.csc_array((data, (row_indices_cache, col_indices_cache)), shape=(2**N, 2**N), dtype=complex)
        return result
    else:
        raise ValueError("Index n must fulfill 0 <= n < N")
    
def sparse_ladder_inc(n, N, row_indices_cache=None, col_indices_cache=None):
    """Returns a CSC sparse matrix representation of the pauli_+ matrix acting over qubit n in a Hilbert space of N qubits
       0 <= n < N"""
    if 0 <= n < N:
        if (row_indices_cache is None) or (col_indices_cache is None):
            row_indices_cache, col_indices_cache = sparse_non_diag_paulis_indices(n, N)
        return (sparse_pauli_x(n, N) + 1j*sparse_pauli_y(n, N))/2
    else:
        raise ValueError("Index n must fulfill 0 <= n < N")
    
def sparse_ladder_dec(n, N, row_indices_cache=None, col_indices_cache=None):
    """Returns a CSC sparse matrix representation of the pauli_- matrix acting over qubit n in a Hilbert space of N qubits
       0 <= n < N"""
    if 0 <= n < N:
        if (row_indices_cache is None) or (col_indices_cache is None):
            row_indices_cache, col_indices_cache = sparse_non_diag_paulis_indices(n, N)
        return (sparse_pauli_x(n, N) - 1j*sparse_pauli_y(n, N))/2
    else:
        raise ValueError("Index n must fulfill 0 <= n < N")

def sparse_pauli_z(n, N):
    """Returns a CSC sparse matrix representation of the pauli_z matrix acting over qubit n in a Hilbert space of N qubits
       0 <= n < N"""
    if 0 <= n < N:
        block_length = 2**(N - n)
        nblocks = 2**n
        block = np.ones(block_length, dtype=int)
        block[block_length//2::] = -1
        diag = np.tile(block, nblocks)
        row_col_indices = np.arange(2**N, dtype=int)
        result = sparse.csc_array((diag, (row_col_indices, row_col_indices)), shape=(2**N, 2**N), dtype=complex)
        return result
    else:
        raise ValueError("Index n must fulfill 0 <= n < N")

# ---------------------------------------------------------------------
#                     PAULI ALGEBRA ABSTRACTIONS
# ---------------------------------------------------------------------

from sympy.physics.paulialgebra import Pauli, evaluate_pauli_product
from sympy.core.singleton import Singleton
from sympy import I, Mul, Add
from sympy.core.numbers import One
from copy import copy

class PauliTensor:
    def __init__(self, factors, coefficient=1):
        coefficients = []
        paulis = []
        for factor in factors:
            if not isinstance(factor, (Pauli, Mul, One, Singleton)):
                raise ValueError("factors must be Sympy Pauli objects")
            if isinstance(factor, (One, Singleton)):
                paulis.append(One)
            else:
                prodels = factor.as_coeff_Mul(rational=True)
                coefficients.append(prodels[0])
                paulis.append(prodels[1])
        self.coefficient = np.prod(coefficients)*coefficient
        if self.coefficient.is_integer:
            self.coefficient = int(self.coefficient)
        self.paulis = paulis
        self.nqubits = len(factors)

    @classmethod
    def from_string(cls, string):
        string_to_op_dict = {"I": One, "X": Pauli(1), "Y":Pauli(2), "Z":Pauli(3)}
        chars = list(string)
        paulis = []
        coeff = 1
        for char in chars:
            if char not in string_to_op_dict.keys():
                try:
                    if char == "i":
                        this_coeff = I
                    elif char == "--":
                        this_coeff = -1
                    else:
                        this_coeff = int(char)
                    coeff *= this_coeff
                except ValueError:
                    raise ValueError("Pauli string must only contain ints, i, -, or the following characters [\"I\", \"X\", \"Y\", \"Z\"]")
            else:
                paulis.append(string_to_op_dict[char])
        return cls(paulis, coeff)
    
    def __mul__(self, other):
        if not isinstance(other, (int, float, complex, PauliTensor)):
            raise TypeError(f"can't multiply PauliTensor by element of type \'{type(other)}\'")
        elif isinstance(other, (int, float, complex)):
            final_coefficient = self.coefficient * other
            new_prod = copy(self)
            new_prod.coefficient = final_coefficient
            return new_prod
        elif isinstance(other, PauliTensor):
            if len(self) != len(other):
                raise TypeError("can't multiply PauliTensor with different dimension")
            new_paulis = []
            for self_pauli, other_pauli in zip(self.paulis, other.paulis):
                if self_pauli is One:
                    this_new_pauli = other_pauli
                elif other_pauli is One:
                    this_new_pauli = self_pauli
                else:
                    this_new_pauli = evaluate_pauli_product(self_pauli*other_pauli)
                new_paulis.append(this_new_pauli)
            # new_paulis = [evaluate_pauli_product(self_pauli*other_pauli) for self_pauli, other_pauli in zip(self.paulis, other.paulis)]
            imaginary_units = 1
            extra_coefficients = 1
            for i, pauli in enumerate(new_paulis):
                if isinstance(pauli, Mul):
                    coeff_product = pauli.as_two_terms()
                    extra_coefficients *= coeff_product[0]
                    try:
                        new_paulis[i] = coeff_product[1].as_two_terms()[1]
                        imaginary_units *= coeff_product[1].as_two_terms()[0]
                    except AttributeError:
                        new_paulis[i] = coeff_product[1]
            new_coefficient = self.coefficient * other.coefficient * extra_coefficients * imaginary_units
            if new_coefficient == 0:
                return PauliTensor([], new_coefficient)
            else:
                return PauliTensor(new_paulis, new_coefficient)
        
    def __rmul__(self, other):
        if not isinstance(other, (int, float, complex, PauliTensor)):
            raise TypeError(f"can't multiply PauliTensor by element of type \'{type(other)}\'")
        elif isinstance(other, (int, float, complex)):
            return PauliTensor.__mul__(self, other)
        
    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            return PauliTensorSum(self, PauliTensor([One]*len(self), coefficient=other))
        elif isinstance(other, PauliTensor):
            return PauliTensorSum(self, other)
        else:
            raise TypeError(f"can't sum PauliTensor and element of type \'{type(other)}\'")
        
    def __radd__(self, other):
        if isinstance(other, (int, float, complex)):
            return PauliTensorSum(self, PauliTensor([One]*len(self), coefficient=other))
        else:
            raise TypeError(f"can't sum PauliTensor and element of type \'{type(other)}\'")

    def __sub__(self, other):
        if isinstance(other, (int, float, complex)):
            return PauliTensorSum(self, PauliTensor([One]*len(self), coefficient=-other))
        elif isinstance(other, PauliTensor):
            negother = PauliTensor(other.paulis, -other.coefficient)
            return PauliTensorSum(self, negother)
        else:
            raise TypeError(f"can't substract PauliTensor and element of type \'{type(other)}\'")
        
    def __rsub__(self, other):
        if isinstance(other, (int, float, complex)):
            return PauliTensorSum(PauliTensor([One]*len(self), coefficient=other), PauliTensor(self.paulis, -self.coefficient))
        else:
            raise TypeError(f"can't sum PauliTensor and element of type \'{type(other)}\'")

    def __len__(self):
        return self.nqubits

    def __repr__(self):
        pauli_str = {One:"I", Pauli(1):"X", Pauli(2):"Y", Pauli(3):"Z"}
        if isinstance(self.coefficient, Mul):
            if self.coefficient.as_two_terms()[1] == I:
                if (c := self.coefficient.as_two_terms()[0]) == -1:
                    coeff_str = "-i*"
                else:
                    coeff_str = f"{c}i*"
            else:
                coeff_str = f"{self.coefficient}*"
        elif isinstance(self.coefficient, Add):
            coeff_str = f"({self.coefficient})*"
        else:
            if self.coefficient == -1:
                coeff_str = "-"
            elif self.coefficient == 1:
                coeff_str = ""
            else:
                coeff_str = str(self.coefficient)
        coeff_str = coeff_str.rstrip("0")
        if self.coefficient == 0:
            coeff_str = "0"
        try:
            if coeff_str[-1] =="*":
                coeff_str = coeff_str[0:-1]
        except IndexError:
            pass
        if self.is_identity():
            pauli_strs = [""]
        else:
            pauli_strs = [pauli_str[pauli] for pauli in self.paulis]
        return f"".join([coeff_str] + pauli_strs)
    
    def __lt__(self, other):
        if not isinstance(other, PauliTensor):
            raise TypeError(f"can't compare PauliTensor with element of type \'{type(other)}\'")
        if self.paulis == other.paulis:
            return self.coefficient < other.coefficient
        else:
            return False 

    def __leq__(self, other):
        if not isinstance(other, PauliTensor):
            raise TypeError(f"can't compare PauliTensor with element of type \'{type(other)}\'")
        if self.paulis == other.paulis:
            return self.coefficient <= other.coefficient
        else:
            return False
        
    def __gt__(self, other):
        if not isinstance(other, PauliTensor):
            raise TypeError(f"can't compare PauliTensor with element of type \'{type(other)}\'")
        if self.paulis == other.paulis:
            return self.coefficient > other.coefficient
        else:
            return False
        
    def __geq__(self, other):
        if not isinstance(other, PauliTensor):
            raise TypeError(f"can't compare PauliTensor with element of type \'{type(other)}\'")
        if self.paulis == other.paulis:
            return self.coefficient > other.coefficient
        else:
            return False

    def __eq__(self, other):
        if not isinstance(other, PauliTensor):
            raise TypeError(f"can't compare PauliTensor with element of type \'{type(other)}\'")
        return (self.coefficient == other.coefficient == 0) or (self.paulis == other.paulis) and (self.coefficient == other.coefficient)
    
    def has_same_paulis(self, other):
        if not isinstance(other, PauliTensor):
            raise TypeError(f"can't compare PauliTensor with element of type \'{type(other)}\'")
        return self.paulis == other.paulis
    
    def is_identity(self):
        return all([pauli == One for pauli in self.paulis])
    
class PauliTensorSum:
    def __init__(self, *products):
        if any([not isinstance(product, PauliTensor) for product in products]):
            raise TypeError("all products must be of type PauliTensor")
        self.nqubits = None
        self.summands = []
        for product in products:
            if not isinstance(product, PauliTensor):
                raise TypeError("all products must be of type PauliTensor")
            if len(product) != 0 and self.nqubits is None:
                self.nqubits = len(product)
                self.summands = [product]
            elif len(product) != 0:
                if len(product) != self.nqubits:
                    raise ValueError("all products must have the same dimension")
                else:
                    for i, present_product in enumerate(self.summands):
                        if present_product.has_same_paulis(product):
                            sumcoeff =  present_product.coefficient + product.coefficient
                            if sumcoeff != 0:
                                self.summands[i] = PauliTensor(present_product.paulis, present_product.coefficient + product.coefficient)
                            else:
                                self.summands[i] = PauliTensor([], 0)
                            break
                    else:
                        self.summands.append(product)
        if len(self.summands) == 0:
            self.summands = [PauliTensor([], 0)]

    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            return PauliTensorSum(*self.summands, PauliTensor([One], coefficient=other))
        elif isinstance(other, PauliTensor):
            return PauliTensorSum(*self.summands, other)
        elif isinstance(other, PauliTensorSum):
            return PauliTensorSum(*self.summands, *other.summands)
        else:
            raise TypeError(f"can't sum PauliTensorSum and an element of type \'{type(other)}\'")
        
    def __radd__(self, other):
        if isinstance(other, (int, float, complex)):
            return PauliTensorSum(*self.summands, PauliTensor([One], coefficient=other))
        elif isinstance(other, PauliTensor):
            return PauliTensorSum(*self.summands, other)
        else:
            raise TypeError(f"can't sum PauliTensorSum and an element of type \'{type(other)}\'")
        
    def __sub__(self, other):
        if isinstance(other, (int, float, complex)):
            return PauliTensorSum(*self.summands, PauliTensor([One], coefficient=-other))
        elif isinstance(other, PauliTensor):
            return PauliTensorSum(*self.summands, other)
        elif isinstance(other, PauliTensorSum):
            return PauliTensorSum(*self.summands, *[PauliTensor(summand.paulis, -summand.coefficient) for summand in other.summands])
        else:
            raise TypeError(f"can't substract PauliTensorSum and element of type \'{type(other)}\'")
        
    def __rsub__(self, other):
        negself = PauliTensorSum(*[PauliTensor(summand.paulis, -summand.coefficient) for summand in self])
        if isinstance(other, (int, float, complex)):
            return PauliTensorSum(PauliTensor([One], other), *negself.summands)
        elif isinstance(other, PauliTensor):
            return PauliTensorSum(other, negself)
        elif isinstance(other, PauliTensorSum):
            return PauliTensorSum(*other.summands, *negself.summands)
        else:
            TypeError(f"can't substract PauliTensorSum and element of type \'{type(other)}\'")

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return PauliTensorSum(*[PauliTensor(summand.paulis, summand.coefficient*other) for summand in self.summands])
        elif isinstance(other, PauliTensor):
            products = []
            for summand in self.summands:
                products.append(summand*other)
            return PauliTensorSum(*products)
        elif isinstance(other, PauliTensorSum):
            product = []
            for selfsummand in self.summands:
                for othersummand in other.summands:
                    product.append(selfsummand*othersummand)
            return PauliTensorSum(*product)
        else:
            TypeError(f"can't substract PauliTensorSum and element of type \'{type(other)}\'")

    def norm_upper_bound(self):
        return np.sum([np.abs(summand.coefficient) for summand in self.summands])
    
    def __repr__(self):
        strings_to_join = [str(self.summands[0])]
        for summand in self.summands[1::]:
            summand_str = str(summand)
            if summand_str[0] == "-":
                joiner = " - "
                summand_str = summand_str[1::]
            else:
                joiner = " + "
            strings_to_join += [joiner, summand_str]
        return "".join(strings_to_join)
    
def commutator(pauli1, pauli2):
    if not (isinstance(pauli1, (Pauli, PauliTensor, PauliTensorSum)) or isinstance(pauli2, (Pauli, PauliTensor, PauliTensorSum))):
        raise TypeError("commutator aguments must be Pauli, PauliTensor or PauliTensorSum")
    if isinstance(pauli1, Pauli):
        paulisum1 = PauliTensorSum(PauliTensor(pauli1))
    elif isinstance(pauli1, PauliTensor):
        paulisum1 = PauliTensorSum(pauli1)
    else:
        paulisum1 = pauli1
    if isinstance(pauli2, Pauli):
        paulisum2 = PauliTensorSum(PauliTensor(pauli2))
    elif isinstance(pauli2, PauliTensor):
        paulisum2 = PauliTensorSum(pauli2)
    else:
        paulisum2 = pauli2

    commutator_sum = []
    for p1 in paulisum1.summands:
        for p2 in paulisum2.summands:
            commutator_sum.append(p1*p2 - p2*p1)
    
    return PauliTensorSum(*[result.summands[0] for result in commutator_sum])