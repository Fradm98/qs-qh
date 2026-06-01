import numpy as np

def constant(lattice, nlayers, angle):
    return np.full((len(lattice.node_coords), nlayers), angle)

def counteract_rotation(lattice, nlayers, ham_param, dt):
    return np.full((len(lattice.node_coords), nlayers), ham_param*dt)

def random(lattice, nlayers):
    return np.random.uniform(-np.pi, np.pi, (len(lattice.node_coords), nlayers))

def zero_mean(lattice, nlayers):
    matrix = np.random.uniform(-np.pi, np.pi, (len(lattice.node_coords), nlayers))
    row_means = matrix.mean(axis=1)
    matrix = matrix - row_means.reshape(-1, 1)
    return matrix