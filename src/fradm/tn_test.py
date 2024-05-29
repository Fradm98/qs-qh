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
    
    def local_order_param(self, site: int, long: str="Z"):
        X = sparse_pauli_x(n=0,L=1).toarray()
        Z = sparse_pauli_z(n=0,L=1).toarray()

        if long == "Z":
            op = Z
        elif long == "X":
            op = X

        self.Z2.mpo_skeleton(aux_dim=2)
        mpo = self.Z2.mpo
        mpo_tot = []
        for i in range(self.L):
            if i == (site):
                mpo[0,-1] = op
            mpo_tot.append(mpo)
            self.Z2.mpo_skeleton(aux_dim=2)
            mpo = self.Z2.mpo
        self.w = mpo_tot
        return self
    
    def TEBD_Z2_chain(self, params_quench):

        # initialize ancilla with a state
        self._compute_norm(site=1)
        self.ancilla_sites = self.sites.copy()

        errors = [[0, 0]]
        entropies = [0
                     ]
        trotter_steps = params_quench.get('trotter_steps')
        delta = params_quench.get('delta')
        h_1 = params_quench.get('h_1')
        h_2 = params_quench.get('h_2')
        n_sweeps = params_quench('n_sweeps')
        conv_tol = params_quench('conv_tol')
        bond = params_quench('bond')
        where = params_quench('where')
        for trott in range(trotter_steps):
            print(f"------ Trotter steps: {trott} -------")
            self.mpo_quench(delta, h_1, h_2)
            print(f"Bond dim ancilla: {self.ancilla_sites[self.L//2].shape[0]}")
            print(f"Bond dim site: {self.sites[self.L//2].shape[0]}")
            error, entropy = self.compression(
                trunc_tol=False,
                trunc_chi=True,
                n_sweeps=n_sweeps,
                conv_tol=conv_tol,
                bond=bond,
                where=where,
            )
            self.ancilla_sites = self.sites.copy()
            errors.append(error)
            entropies.append(entropy)