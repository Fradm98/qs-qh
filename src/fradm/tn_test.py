from qs_mps.mps_class import MPS
from qs_mps.sparse_hamiltonians_and_operators import sparse_pauli_z, sparse_pauli_x, sparse_pauli_y
from qs_mps.utils import tensor_shapes, mpo_to_matrix

import numpy as np
from ncon import ncon
from scipy.sparse import csc_array, identity, linalg

from fradm.utils import pauli_decomposition

# we can subclass the mps to a specific one
class Z2_chain_massive_mps(MPS):
    def __init__(self, L, d, model="Z2_chain_massive", chi=None, J=None, h1=None, h2=None):
        """
        
        """
        super().__init__(L, d, model, chi)
        self.J = J
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
        c1 = [self.J,self.h1]
        c2 = [1,0]
        for i in range(self.L):
            w = np.array(
                [[I, c2[i%2] * X, O, -c1[i%2] * Z],
                 [O, O, X, O],
                 [O, O, O, -self.h2 * c2[i%2] * X],
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
    
    def interaction_trotter_mpo(self, delta):
        I = identity(2).toarray()
        O = np.zeros((2,2))
        X = sparse_pauli_x(0,1).toarray()
        w_start = np.asarray([(np.cos(delta*self.h2))**(1/3)*I, (np.sin(delta*self.h2))**(1/3)*X]).reshape((1,2,2,2))
        w_middle = np.asarray([[(np.cos(delta*self.h2))**(1/3)*I, O],[O, (np.sin(delta*self.h2))**(1/3)*X]])
        w_end = np.asarray([(np.cos(delta*self.h2))**(1/3)*I, 1j*(np.sin(delta*self.h2))**(1/3)*X]).reshape((2,1,2,2))
        w_se = ncon([w_start,w_end],[[-1,-3,-5,1],[-2,-4,1,-6]]).reshape((2,2,2,2))
        w_bulk = [w_se,w_middle]
        w_tot = []
        for i in range(self.L):
            if i == 0:
                w = w_start
            elif i == 1:
                w = w_middle
            elif i > 1 and i < (self.L-2):
                w = w_bulk[i%2]
            elif i == (self.L-2):
                w = w_middle
            elif i == (self.L-1):
                w = w_end
            w_tot.append(w)
        self.w = w_tot
        return self
    
    def local_trotter_mpo(self,delta):
        Z = sparse_pauli_z(n=0,L=1).toarray()
        # divide over 2 to make the second order trotter decomposition
        c = [self.J/2,self.h1/2]
        w_tot = [linalg.expm(1j*c[i%2]*delta*Z) for i in range(self.L)]
        return w_tot

    def second_order_trotter_mpo(self, delta):
        w_loc = self.local_trotter_mpo(delta)
        self.interaction_trotter_mpo(delta)
        w_tot = [ncon([w_loc[i],self.w[i],w_loc[i]],[[-3,1],[-1,-2,1,2],[2,-4]]) for i in range(self.L)]
        self.w = w_tot
        return self

    def TEBD_Z2_chain(self, params_quench):

        # initialize ancilla with a state
        self._compute_norm(site=1)
        self.ancilla_sites = self.sites.copy()

        errors = [[0, 0]]
        entropies = [[0]]
        schmidt_vals = []
        exp_vals = []
        trotter_steps = params_quench.get('trotter_steps')
        delta = params_quench.get('delta')
        n_sweeps = params_quench.get('n_sweeps')
        conv_tol = params_quench.get('conv_tol')
        bond = params_quench.get('bond')
        where = params_quench.get('where')
        if where == -1:
            where = self.L//2

        # Exp val before trotterization
        for i in range(self.L):
            self.local_order_param(site=i)
            exp_vals.append(self.mpo_first_moment().real)

        for trott in range(trotter_steps):
            print(f"------ Trotter steps: {trott} -------")
            self.second_order_trotter_mpo(delta)
            print(f"Bond dim ancilla: {self.ancilla_sites[self.L//2].shape[0]}")
            print(f"Bond dim site: {self.sites[self.L//2].shape[0]}")
            error, entropy, s_mid = self.compression(
                trunc_tol=False,
                trunc_chi=True,
                n_sweeps=n_sweeps,
                conv_tol=conv_tol,
                bond=bond,
                where=where,
            )
            self.ancilla_sites = self.sites.copy()

            # Exp val during trotterization
            for i in range(self.L):
                self.local_order_param(site=i)
                exp_vals.append(self.mpo_first_moment().real)
           
            errors.append(error)
            entropies.append(entropy)
            schmidt_vals.append(s_mid)

        exp_vals = np.array(exp_vals).reshape((trotter_steps+1,self.L))
        return errors, entropies, schmidt_vals, exp_vals