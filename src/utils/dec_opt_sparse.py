# ------------------------------------------------- #
#        Distribution Error Correction (DEC)        #
#     (Optimizations & corrections on filling)      #
#         https://arxiv.org/abs/2501.01953          #
# ------------------------------------------------- #

from juliacall import Main as jl
from sklearn.linear_model import OrthogonalMatchingPursuit

def first_col_update_label(col_label, key):
    num_qubits=len(col_label)
    return bin(int(col_label, 2) ^ int(key, 2))[2:].zfill(num_qubits)

def convert_to_first_col(counts: dict, col_label: str):
    '''The label is the associated column of tilde A. Counts is the associated column values.
    We should convert these to the first column by the symmetry rule described in the paper.'''
    counts={first_col_update_label(col_label, key):value for (key, value) in counts.items()}
    return counts