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
