import numpy as np

def constant(lattice, dt, step, angle):
    return np.full(len(lattice.node_coords), angle)

def counteract_rotation(lattice, dt, step, ham_param):
    return np.full(len(lattice.node_coords), ham_param*dt)

def random(lattice, dt, step):
    return np.random.uniform(-np.pi, np.pi, len(lattice))

def from_matrix(lattice, dt, step, matrix):
    matrix = np.array(matrix)
    if matrix.shape[0] != len(lattice.node_coords):
        raise ValueError("matrix number of rows must match the number of nodes in the heavy hex lattice")
    if step > matrix.shape[1] - 1:
        raise ValueError("step must be less than number of columns in matrix")
    return matrix[:, step]

def zero_mean_gdd_matrix(lattice, steps):
    matrix = np.random.uniform(-np.pi, np.pi, (len(lattice.node_coords), steps))
    row_means = matrix.mean(axis=1)
    matrix = matrix - row_means.reshape(-1, 1)
    return matrix