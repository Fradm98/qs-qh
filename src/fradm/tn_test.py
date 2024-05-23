from qs_mps.mps_class import MPS
from qs_mps.sparse_hamiltonians_and_operators import sparse_pauli_z, sparse_pauli_x, sparse_pauli_y
from qs_mps.utils import tensor_shapes, mpo_to_matrix

import numpy as np
from ncon import ncon
from scipy.sparse import csc_array, identity, linalg

from fradm.utils import pauli_decomposition

# we can subclass the mps to a specific one
class Z2_chain_massive_mps(MPS):
    def __init__(self, L, d, model="Z2_chain_massive", chi=None, h1=None, h2=None, w=None):
        super().__init__(L, d, model, chi, w)
        self.h1 = h1
        self.h2 = h2
        
    def mpo(self, long="Z", trans="X"):
        if self.model == "Z2_chain_massive":
            self.Z2_chain_massive_mpo()

    def Z2_chain_massive_mpo(self):
        I = identity(self.d, dtype=complex).toarray()
        O = csc_array((self.d, self.d), dtype=complex).toarray()
        X = sparse_pauli_x(n=0,L=1).toarray()
        Z = sparse_pauli_z(n=0,L=1).toarray()
        w_tot = []
        for i in range(self.L):
            if (i % 2) == 0:
                c1 = 1
                c2 = 1
            else:
                c1 = self.h1
                c2 = 0
            w = np.array(
                [[I, c2 * X, O, -c1 * Z],
                 [O, O, X, O],
                 [O, O, O, -self.h2 * c2 * X],
                 [O, O, O, I]]
            )
            w_tot.append(w)
        self.w = w_tot
        return self
    
chain = Z2_chain_massive_mps(L=61, d=2, chi=64, h1=0.5, h2=0.1)
chain._random_state(seed=3, chi=16)
chain.canonical_form()
chain.DMRG(trunc_tol=False, trunc_chi=True)
# chain.DMRG(trunc_tol=True, trunc_chi=False)
tensor_shapes(chain.sites)

# chain.mpo_first_moment()