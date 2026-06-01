import numpy as np

def expectation(state, operator):
    return np.conj(state) @ operator @ state