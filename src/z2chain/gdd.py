import numpy as np

def constant(chain_length, nlayers, angle):
    """Returns angles for GDD in constant fashion"""
    return np.full((chain_length, nlayers), angle)

def counteract_rotation(chain_length, nlayers, ham_param, dt):
    return np.full((chain_length, nlayers), ham_param*dt)

def random(chain_length, nlayers):
    """Returns angles for GDD in random fashion"""
    return np.random.uniform(-np.pi, np.pi, (chain_length, nlayers))

def zero_mean(chain_length, nlayers):
    matrix = np.random.uniform(-np.pi, np.pi, (chain_length, nlayers))
    row_means = matrix.mean(axis=1)
    matrix = matrix - row_means.reshape(-1, 1)
    return matrix