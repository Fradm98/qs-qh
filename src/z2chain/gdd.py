import numpy as np

def constant(chain_length, dt, step, angle):
    """Returns angles for GDD in constant fashion"""
    return np.full(chain_length, angle)

def counteract_rotation(chain_length, dt, step, ham_param):
    return np.full(chain_length, ham_param*dt)

def random(chain_length, dt, step):
    """Returns angles for GDD in random fashion"""
    return np.random.uniform(-np.pi, np.pi, chain_length)

def from_matrix(chain_length, dt, step, matrix):
    matrix = np.array(matrix)
    if matrix.shape[0] != chain_length:
        raise ValueError("matrix number of rows must match chain length")
    if step > matrix.shape[1] - 1:
        raise ValueError("step must be less than number of columns in matrix")
    return matrix[:, step]

def zero_mean_gdd_matrix(chain_length, steps):
    matrix = np.random.uniform(-np.pi, np.pi, (chain_length, steps))
    row_means = matrix.mean(axis=1)
    matrix = matrix - row_means.reshape(-1, 1)
    return matrix
