from scipydirect import minimize
from operator import mul
from functools import reduce

import warnings
import collections
import scipy.io as sio

from qutip import*
import numpy as np
import scipy as sp
from copy import copy
from copy import deepcopy
from scipy.sparse.linalg import expm_multiply
from math import *
import matplotlib.pyplot as plt
import math
import cmath
import numbers
import sys
import functools
from scipy.linalg import expm
from numpy.linalg import norm
from scipy.integrate import ode
from qutip.solver import Options
from qutip.cy.spmath import zcsr_trace, zcsr_adjoint, zcsr_mult, zcsr_kron
from qutip.cy.spconvert import zcsr_reshape
# from libs.import_notebook import *
from contract_fast.contract_fast import cy_contract

DO_PARALLEL = True
from funmv_pylib import *

try:
    from correlators_compute_copy.corr_ssh_comp import get_SSH_correlators_old
    from correlators_compute.corr_ssh_comp import get_correlators as get_correlators_c
    from correlators_compute.corr_ssh_comp import get_e1_and_e0 as get_e1_and_e0_c
except:
    pass

import matplotlib

# import importlib
# import funmv
# importlib.reload(funmv)
# from funmv import *

import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
def mkl_set_num_threads(cores):
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))
inch = 3.38583 + 0.2
fontsize = 10
fontsize_Numbers = 8

font = {'family' : 'serif',
       'serif'  : ['STIXGeneral'],
       #'sans-serif':['Helvetica'],
       'weight' : 'regular',
       'size'   : fontsize}

dpi = 72
plt.rc('font', **font)
plt.rc('text', usetex=True)
colors_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
def printNum(current_val, min_val, glob_min_val, file_logs = None):
    glob_min_val_str = glob_min_val if isinstance(glob_min_val, str) else "{:.5f}".format(glob_min_val)
    message = '(' + "{:.5f}".format(current_val) + ') '+"{:.5f}".format(min_val) + " / " + glob_min_val_str
    
    message = '\r%s' % message
    my_print(message, file_name = file_logs )
    sys.stdout.write( message )
    sys.stdout.flush()
    
    sys.__stdout__.write(message)
    sys.__stdout__.flush()

def my_print(text, update = False, file_name = None):
    
    if file_name is None:
        if update:
            text = '\r%s' % text
        else:
            text+='\n'
        sys.stdout.write( text)
        sys.stdout.flush()
    else:
        with open(file_name, "a") as text_file:
#             text_file.write('\n'+text)
            print(text, file=text_file)
# Variational constant
eps = 1e-8

# Basis spin 1/2
a0 = fock(2,0)
a1 = fock(2,1)

# Pauli
S_x = sigmax()
S_y = sigmay()
S_z = -sigmaz()
S_m = sigmap()
S_p = sigmam()

I = identity(2)
# spin 1
rm = fock(3,0)
r0 = fock(3,1)
rp = fock(3,2)

Sp0 = rp* r0.dag()
S0m = r0* rm.dag()
Sp0_0m = Sp0 + S0m

Sr_x = (Sp0_0m + Sp0_0m.dag())/2**0.5
Sr_y = 1j*(-Sp0_0m + Sp0_0m.dag())/2**0.5
Sr_z = rp*rp.dag() - rm*rm.dag()

# map from 2 Rydebergs, spins 1/2, to spin 1
map_01 = tensor(a0,a1)
map_00 = tensor(a0,a0)
map_10 = tensor(a1,a0)

trans_map_ryd = tensor(rm,map_01.dag()) + tensor(r0,map_00.dag()) + tensor(rp,map_10.dag())
trans_map_ryd.dims = [[3],[2,2]]

# map from 2 ions, spins 1/2, to spin 1
# rm = fock(4,0)
# r0 = fock(4,1)
# rp = fock(4,2)
# re = fock(4,3)

# Sp0 = rp* r0.dag()
# S0m = r0* rm.dag()
# Sp0_0m = Sp0 + S0m
# Se = re*re.dag()


# Sr_x = (Sp0_0m + Sp0_0m.dag())/2**0.5
# Sr_y = 1j*(-Sp0_0m + Sp0_0m.dag())/2**0.5
# Sr_z = rp*rp.dag() - rm*rm.dag()

# map_00 = tensor(a0,a0)
# map_10 = tensor(a1,a0)
# map_01 = tensor(a0,a1)
# map_11 = tensor(a1,a1)
# map_10p01 = (tensor(a0,a1) + tensor(a1,a0)).unit()
# map_10m01 = (tensor(a0,a1) - tensor(a1,a0)).unit()

# trans_dims = [[4],[2,2]]

# trans_map_ryd = tensor(rm,map_01.dag()) + tensor(r0,map_00.dag()) + tensor(rp,map_10.dag())
# trans_map_ryd.dims = trans_dims

# trans_map_ion = tensor(rm,map_00.dag()) + tensor(r0,map_01.dag()) + tensor(rp,map_11.dag())
# # trans_map_ion = tensor(rm,map_00.dag()) + tensor(r0,map_10p01.dag()) + tensor(rp,map_11.dag()) + tensor(re,map_10m01.dag())
# # trans_map_ion = tensor(rm,map_00.dag()) + tensor(r0,map_10p01.dag()) + tensor(rp,map_11.dag())
# # trans_map_ion.dims = [[3],[2,2]]
# trans_map_ion.dims = trans_dims

# # Mixed blockaded state of twp Rydberg atoms
# I_block = Qobj(np.diag([1,1,1,0]),dims = [[2,2],[2,2]])

FORMAT_IDENT = 'ident'
FORMAT_DIAG = 'diag'
FORMAT_SPARSE = 'sparse'
FORMAT_DENSE = 'dense' 
FORMAT_SPARSE_REAL_IMAG = 'sparse_real_imag'

TYPE_STATE = 'state'
TYPE_HAMILTONIAN = 'Hamiltonian'
TYPE_OPERATOR = 'operator'

class MyQobj(object):
    '''
    Object represented quantum states and operators in defferent formats.
    The object takes care about product of different formats. 
    
    Parameters
    ----------
    data : one of following forms
    
        FORMAT_IDENT: 1
        FORMAT_DIAG: array
            Diagonal of the matrix
        FORMAT_SPARSE: matrix sparse csr
        FORMAT_DENSE: matrix array
        FORMAT_SPARSE_REAL_IMAG: [sparse csr, sparse csr], both with format float64
            Real and imaginary parts of the state.
    
    q_type : str
            
    '''
    
    
    
    def __init__( self, data, q_type = TYPE_OPERATOR, is_super = False, dims = None):
        
        self.is_super = is_super
        self.q_type = q_type
        
        self.set_data(data, dims)

    def set_data(self, data, dims):

        if isinstance(data, Qobj):
#             dims = data.dims[0]
            dims = data.dims
            data = data.data

        self.data = data
        
        if data is 1:
            self.format = FORMAT_IDENT
            self.complexity = 1
            
        elif isinstance(data, qutip.fastsparse.fast_csr_matrix) or isinstance(data, sp.sparse.csr.csr_matrix):
            self.format = FORMAT_SPARSE
#             self.complexity = np.prod(data.shape)
            
        elif isinstance(data, list):
            self.format = FORMAT_SPARSE_REAL_IMAG
            self.complexity = np.prod(data[0].shape)
            
        elif isinstance(data, np.ndarray):
#             self.complexity = np.prod(data.shape)
            
            if len(data.shape) == 1:
                self.format = FORMAT_DIAG
            else:
                self.format = FORMAT_DENSE
        else:
            raise ValueError
        
        if dims is None:
            shape = self.get_shape()
            dims = [[shape[0]],[shape[1]]]
        
#         for i, D in enumerate(dims):
#             if len(D)>1:
#                 D = [d for d in D if d!=1]
#             if len(D)==0: D = [1]
            
#             dims[i] = D
            
#         print(dims)

#         if len(dims[0]) != len(dims[1]):
#             raise Exception('len(dims[0]) != len(dims[1]):')
            
        self.dims = check_dims(dims)
    
    def to_dense(self, is_copy = False):
        if self.format == FORMAT_SPARSE:
            data = self.data.toarray()
        elif self.format in [FORMAT_DENSE, FORMAT_IDENT]:
            data = self.data
        else:
            raise NotImplementedError
            
        if is_copy:
            return MyQobj(data, self.q_type, is_super = self.is_super, dims = self.dims)
        else:
            self.set_data(data, self.dims)
    
    def to_qobj(self):
#         return Qobj(self.data, dims = [self.dims]*2)
        return Qobj(self.data, dims = self.dims)
    
    def to_sparse(self, is_copy = False):
        if self.format == FORMAT_SPARSE:
            data = self.data
        elif self.format == FORMAT_DENSE:
            data = sp.sparse.csr_matrix(self.data)
#             data = fast_csr_matrix((_tmp.data, _tmp.indices, _tmp.indptr),
#                                          shape=_tmp.shape)
            
        else:
            raise NotImplementedError
            
        if is_copy:
            return MyQobj(data, self.q_type, is_super = self.is_super, dims = self.dims)
        else:
            self.set_data(data, self.dims)
            
    def vector_to_operator(self, is_copy = False):
        # Super vector state to density matrix
        if self.q_type != TYPE_STATE:
            raise Exception("Only for TYPE_STATE")
        
        if self.is_super and len(self.data.shape)==2 and self.data.shape[-1]==1:
            data = my_vec_to_op(self.data)
        else:
            data = self.data

        if is_copy:
            return MyQobj(data, TYPE_STATE, is_super = False, dims = self.dims)
        else:
            self.data = data
            self.is_super = False
        
    def operator_to_vector(self, is_copy = False):
        # Density matrix to super vector state
        if self.q_type != TYPE_STATE:
            raise Exception("Only for TYPE_STATE")
        
        if self.format not in [FORMAT_DENSE, FORMAT_SPARSE]:
            raise Exception('self.format = '+self.format)
        
        data = self.data
        if not self.is_super:
            if self.data.shape[1] == 1:
                data = data * data.conj().T
            data = my_op_to_vec(data)
        
        if is_copy:
            return MyQobj(data, TYPE_STATE, is_super = True, dims = self.dims)
        else:
            self.data = data
            self.is_super = True
    
    def to_super(self, is_copy = False):
        q_type = self.q_type
        
        # Apply sprepost to data
        if self.format == FORMAT_IDENT or self.is_super:
            data = self.data
            
        elif self.format == FORMAT_DIAG:
            
            if q_type == TYPE_HAMILTONIAN:
                data = to_super_H_diag(self.data)
            elif q_type == TYPE_OPERATOR:
                data = to_super_oper_diag(self.data)
            else:
                raise NotImplementedError
            
        elif self.format in [FORMAT_DENSE, FORMAT_SPARSE]:
            
            if q_type == TYPE_HAMILTONIAN:
                data = to_super_H(self.data)
            elif q_type == TYPE_OPERATOR:
                data = to_super_oper(self.data)
            elif q_type == TYPE_STATE:
                return self.operator_to_vector(is_copy)

            if self.format == FORMAT_DENSE and not isinstance(data,np.ndarray) :
                data = data.toarray()

        else:
            raise NotImplementedError
        
        dims = self.dims
        if not self.is_super:
            dims = [[dims[0]]*2,[dims[1]]*2]
            
        if is_copy:
            return MyQobj(data, q_type, is_super = True, dims = dims)
        else:
            self.dims = dims
            self.data = data
            self.is_super = True
    
    def tr(self):
        if self.format == FORMAT_IDENT or self.is_super:
            tr = 1
        elif self.format == FORMAT_DIAG:
            tr = sum(self.data)
        elif self.format == FORMAT_DENSE or FORMAT_SPARSE:
            tr = sum(self.data.diagonal())
        else:
            raise NotImplementedError
        
        return tr
        
        
    def dag(self):
        data = self.data
        
        if self.format == FORMAT_IDENT:
            pass
        elif self.format == FORMAT_DIAG:
            data = data.conj().T

        elif self.format == FORMAT_DENSE:
            data = data.conj().T.copy()

        elif self.format == FORMAT_SPARSE:
            data = zcsr_adjoint(data)

        else:
            raise NotImplementedError
        
        return MyQobj(data, self.q_type, is_super = self.is_super, dims = self.dims[::-1])
        
    def __div__(self, other):
        if isinstance(other, (int, np.int64, float, complex)):
            return MyQobj(self.data / other, self.q_type, is_super = self.is_super, dims = self.dims)
        else:
            raise ValueError
        
    def __mul__(self, other):

        if isinstance(other, (int, np.int64,float, complex)):
#             if other == 0:
#                 return 0
            return MyQobj(self.data * other, self.q_type, is_super = self.is_super, dims = self.dims)

        elif type(other) != MyQobj:
            raise NotImplementedError
        
        if self.is_super or other.is_super:
            self.to_super()
            other.to_super()
        
        A = self.data
        B = other.data
        
        if self.format == FORMAT_IDENT:
            return other
        elif other.format == FORMAT_IDENT:
            return self
        
        elif self.format == FORMAT_DIAG:
            
            if other.format == FORMAT_DIAG:
                st_out = A * B
            elif other.format == FORMAT_DENSE:
                st_out = (B.T * A).T
            else:
                raise NotImplementedError
            
        elif self.format == FORMAT_DENSE:
            
            if other.format == FORMAT_DIAG:
                st_out = A * B
            elif other.format == FORMAT_DENSE:
                st_out = np.dot(A, B)
            elif other.format == FORMAT_SPARSE:
#                 print(1)
                st_out = A * B
            else:
                raise NotImplementedError
        
        elif self.format == FORMAT_SPARSE:
            if other.format == FORMAT_DENSE:
#                 print(2)
                if B.shape[1] == 1:
                    st_out = my_mv(A, B)
                else:
                    st_out = my_mm(A, B)
                    
#                 st_out = A * B
            elif other.format == FORMAT_SPARSE:
                st_out = A * B
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        if TYPE_STATE in [self.q_type, other.q_type]:
            q_type_out = TYPE_STATE
#         elif TYPE_HAMILTONIAN in [self.q_type, other.q_type]:
#             q_type_out = TYPE_HAMILTONIAN
        else:
            q_type_out = TYPE_OPERATOR
        
        return MyQobj(st_out, q_type_out, is_super = other.is_super, dims = other.dims)
    
    __rmul__ = __mul__
        
    def __add__(self, other):
        
        if self.format == FORMAT_SPARSE_REAL_IMAG:
            raise NotImplementedError
            
        elif isinstance(other, (int, np.int64, float, complex)):
            data = self.data + other
        
        elif self.format != other.format:
            raise NotImplementedError
        else:
            
            if self.q_type != other.q_type:
                raise ValueError("Only the same q_types could be summed up")
            
            if self.is_super or other.is_super:
                self.to_super()
                other.to_super()
                
            data = self.data + other.data
        
        return MyQobj(data, self.q_type, is_super = self.is_super, dims = self.dims)
    
    __radd__ = __add__
    
    def __repr__(self):
        return self.__str__()
#         data_str = self.data.__repr__()
#         return 'MyQobj\nis_super = ' + str(self.is_super) + '\nData:\n' + data_str + '\n'
    
    def __str__(self):
        data_str = str(self.data)
        return 'MyQobj' + '\n' +     'q_type = ' + self.q_type + '\n'+    'is_super = ' + str(self.is_super) + '\n'+    'format = ' + self.format + '\n'+    'shape = ' + str(self.get_shape()) + '\n'+    'dims = ' + str(self.dims) + '\n'+    'Data:'+'\n' +     data_str + '\n\n'
    
    def get_shape(self):
        data = self.data
        
        if self.format == FORMAT_SPARSE_REAL_IMAG:
            shape = data[0].shape
        elif self.format == FORMAT_IDENT:
            shape = [1,1]
        else:
            shape = data.shape
            
        return shape
    
    def expm(self):
        
        if self.q_type == TYPE_STATE:
            raise Exception("TYPE_STATE could not be exponented")
        
        if self.format == FORMAT_DIAG:
            data = np.exp(self.data)
        elif self.format == FORMAT_DENSE:
            data = sp.linalg.expm(self.data)
        elif self.format == FORMAT_SPARSE:
            # Has to be corrected
            data = sp.linalg.expm(self.data.todense())
        else:
            raise NotImplementedError
    
        return MyQobj(data, TYPE_OPERATOR, is_super = self.is_super, dims = self.dims)

        
    def expm_multiply(self, B):
        A = self
        if type(A) != MyQobj or type(B) != MyQobj:
            raise ValueError("A and B have to be MyQobj") 
            
        if A.q_type == TYPE_STATE:
            raise Exception("TYPE_STATE could not be exponented")
            

        if A.is_super or B.is_super:
            A.to_super()
            B.to_super()

        if A.format == FORMAT_IDENT:
            return B
        elif B.format == FORMAT_IDENT:
            return A.expm()
        elif A.format == FORMAT_DIAG:
            return A.expm() * B
        elif A.format == FORMAT_DENSE:
            return A.expm() * B
        elif A.format == FORMAT_SPARSE:
            if B.format == FORMAT_DENSE:
                if DO_PARALLEL:
                    data = expm_multiply_parallel(A.data, B.data)
                else:
                    data = expm_multiply(A.data, B.data)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
            
            
        if B.q_type == TYPE_STATE:
            q_type_put = TYPE_STATE
        else:
            q_type_put = TYPE_OPERATOR
        
        return MyQobj(data, q_type_put, is_super = B.is_super, dims = [A.dims[0], B.dims[1]])
    
#     def tr(self):
#         if self.format is FORMAT_DENSE:
#             return np.trace(self.data)
#         else:
#             return zcsr_trace(self.data, True)

def check_dims(dims):
    dims = deepcopy(dims)
    N_d = len(dims[0])
    if N_d == 1:
        return dims
    
    for i in range(N_d)[::-1]:
        if dims[0][i] == 1 and dims[1][i] == 1:
            del dims[0][i]
            del dims[1][i]
            
    return dims
def tensor_mq(states_list):
    st_out = states_list[0].data
    if states_list[0].format is FORMAT_SPARSE:
        for st in states_list[1:]:
            st_out = zcsr_kron(st_out, st.data)
    else:
        for st in states_list[1:]:
            st_out = np.kron(st_out, st.data)
            

    dims_0 = np.concatenate([st.dims[0] for st in states_list]).ravel().tolist()
    dims_1 = np.concatenate([st.dims[1] for st in states_list]).ravel().tolist()
    dims = [dims_0, dims_1]
    
    return MyQobj(st_out, q_type = states_list[0].q_type, dims = dims)
def my_mv(A, B):
    if DO_PARALLEL:
        st_out = mkl_zspmv(A, B)
    else:
        st_out = A * B
    
    return st_out

def my_mm(A, B):
    if DO_PARALLEL:
        st_out = mkl_zspmm(A, B)
    else:
        st_out = A * B
    
    return st_out
class Optimiz_props(object):
    '''
    Properties of the optimization algorithm.
    
    Parameters
    ----------
    method: str, for example 'BFGS', "basinhop"
        Inserted to sp.optimize.minimize
    tol_rel: float
        Optimization tolerance in E_min/E_max
    maxiter: int
        Max number of ineration of sp.optimize.minimize
    is_kraus: bool
        If True Kraus map is culculated and are applyed to the initial state required times. 
        Use if the map is the same.
        If False the whole evolution is applied to the initial state directly. 
        Use if the initial state is pure and the map is applied ones or twise.
    jac: bool
        If True, the Jacobian for sp.optimize.minimize is calculated analytically during the function evaluation.
        If False, sp.optimize.minimize varies parameters an evaluate function several times.
    do_sparse: bool
        Make calculations with sparse or dense matrices.
    print_data: bool
        Print progress of minimization.
    '''
    
    do_sparse = False
    do_approx = False
    
    MIN_glob = np.inf
    
    def __init__( self, method = 'basinhop', tol_rel = 1e-3, maxiter = 0, jac = True, do_sparse = True, print_data = True, 
                 # next are deprecated
                 N_approx = 5, use_probs = False, P_N_samp = 0, time_dep = False, file_logs = None):
        
        self.do_sparse = do_sparse
        self.method = method 
        self.tol_rel = tol_rel
        self.maxiter = maxiter
        self.print_data = print_data
        self.jac = jac
        self.N_approx = N_approx
        self.use_probs = use_probs
        self.P_N_samp = P_N_samp
        self.time_dep = time_dep
        
        # up to which iteration use probabilities of outcomes
        if use_probs and P_N_samp == 0: self.P_N_samp = N_iter
class System(object):

    '''
    Parameters
    ----------
    state_in : Qobj / [[Qobj, inds_list],...]
                System initial state, where Qobj-s have type 'ket' or 'oper', and inds_list-s are lists of the
                modes indises of the state generated by a tesor product of the list of Qobj-s.
                
    logic_mode_inds: [[mode_ind, lvls_list],...] or [mode_ind,...]
                Logical subspase. List of mode_ind of the state in the correponding order and the indices of levels, 
                lvls_list, of the mode with mode_ind corresponded to {|i>} in the correponding order.
                If logic_mode_inds = None, the whole space is logical. 
                If no lvls_list, a whole mode space is logical
                The logical subspase must fit the simulated model.
                
    inverse_logic: bool
                If True logic_mode_inds -> aux_mode_inds
    '''
    
    def __init__( self, state_in, logic_mode_inds = None, inverse_logic = False):
        
        if isinstance(state_in, Qobj):
            state_in = [[state_in, range(get_q_N_mode(state_in))]]
        elif isinstance(state_in, list):
            if isinstance(state_in[0], Qobj):
                state_in_new = []
                N_modes = 0
                for state in state_in:
                    n_modes = get_q_N_mode(state)
                    state_in_new += [[state, list(range(N_modes, N_modes+n_modes))]]
                    N_modes += n_modes
                state_in = state_in_new
        else:
            raise ValueError()
        
        self.initialize_state(state_in)
        self.initialize_logic_space(logic_mode_inds, inverse_logic)
        
    def initialize_logic_space(self, logic_mode_inds, inverse_logic):
        
        dims_state_list = self.dims_state_list
        
        if logic_mode_inds is None:
            logic_mode_inds = list(range(len(dims_state_list)))
        
        logic_mode_inds_new = []
        for logic_mode in logic_mode_inds:
            if isinstance(logic_mode, int):
                logic_mode = [logic_mode]
            if len(logic_mode) == 1:
                dim_logic_mode = dims_state_list[logic_mode[0]]
                logic_lvls = list(range(dim_logic_mode))
                logic_mode = [logic_mode[0], logic_lvls]
                
            logic_mode_inds_new += [logic_mode]
        
        
        
        logic_mode_inds = logic_mode_inds_new
        aux_mode_inds = []

        logic_modes = [a[0] for a in logic_mode_inds_new]
        logic_lvls = [a[1] for a in logic_mode_inds_new]
        for i, dim in enumerate(dims_state_list):
            if i in logic_modes:
                aux_lvls = [ind for ind in range(dim) if ind not in logic_lvls[logic_modes.index(i)]]
                if len(aux_lvls)>0:
                    aux_mode_inds += [[i, aux_lvls]]
            else:
                aux_lvls = list(range(dim))
                aux_mode_inds += [[i, aux_lvls]]
        
        if inverse_logic:
            logic_mode_inds, aux_mode_inds = aux_mode_inds, logic_mode_inds

        self.N_sys = len(logic_mode_inds)
        self.N_aux = len(aux_mode_inds)
        self.logic_mode_inds = logic_mode_inds
        self.aux_mode_inds = aux_mode_inds
            
    
    def initialize_state(self, state_in_list):
        
        self.state_in_list = state_in_list
        
        # Separate pure and dens states
        inds_pure_part = []
        inds_dens_part = []
        pure_state_list = []
        dens_state_list = []
        
        for qobj, inds_list in state_in_list:
            
            if not isinstance(qobj, Qobj):
                raise ValueError("qobj has to be a Qobj")
            
            # Check inds_list
            if len(inds_list) != get_q_N_mode(qobj):
                raise ValueError("len(inds_list) has to be equal to the modes number of the qobj")
                
            inds_list = list(inds_list)
            
            if qobj.type is 'oper':
                dens_state_list += [qobj]
                inds_dens_part += inds_list
            else:
                pure_state_list += [qobj]
                inds_pure_part += inds_list
        
        # Check inds
        inds_state = sorted(inds_dens_part + inds_pure_part)
        N_modes = max(inds_state)+1
        if inds_state != list(range(N_modes)):
            raise ValueError('Inds of the state must not repeat and must be serial')
        
        
        self.N_modes = N_modes
        self.inds_pure_part = inds_pure_part
        self.inds_dens_part = inds_dens_part
        
        
        # Init tensor states
        if len(pure_state_list)>0:
            self.dims_pure_part_list = list(np.ravel(
                [get_q_dim_list(s) for s in pure_state_list ]
            ))
        else:
            self.pure_state_part = 1
            self.dims_pure_part_list = []
            
        if len(dens_state_list)>0:
            self.dims_dens_part_list = np.concatenate([get_q_dim_list(s) 
                                                       for s in dens_state_list ]).ravel().tolist()
        else:
            self.dens_state_part = 1
            self.dims_dens_part_list = []
            
        # State dims according to the modes order
        self.dims_state_list = [x for _,x in sorted(zip(
            inds_pure_part + inds_dens_part,
        self.dims_pure_part_list + self.dims_dens_part_list,
        ))]
        
        self.pure_state_list = pure_state_list
        self.dens_state_list = dens_state_list

        self.shape = np.prod(self.dims_state_list)
        
        dense_states_full_list = [s[0] if s[0].type is 'oper' else ket2dm(s[0]) for s in self.state_in_list]
        
        self.dense_states_full_list = dense_states_full_list
#         self.dense_states_mq_full_list = [MyQobj(d, q_type = TYPE_STATE, dims = get_q_dim_list(d)) for d in dense_states_full_list]
        self.dense_states_mq_full_list = [MyQobj(d.full(), q_type = TYPE_STATE, dims = d.dims) for d in dense_states_full_list]
        self.inds_list = [s[1] for s in state_in_list]
        
        self.state_mq_in_list = [MyQobj(s[0], q_type = TYPE_STATE) for s in state_in_list]
        
    def init_full_states(self):
        pure_state_list = self.pure_state_list
        dens_state_list = self.dens_state_list
        
        self.pure_state_part = tensor(pure_state_list) if len(pure_state_list)>0 else 1
        self.dens_state_part = tensor(dens_state_list) if len(dens_state_list)>0 else 1
        
        self.dens_state_part_mqobj = MyQobj(self.dens_state_part, q_type = TYPE_STATE)
        self.pure_state_part_mqobj = MyQobj(self.pure_state_part, q_type = TYPE_STATE)
        
    def get_pure_part_in_full_space(self):
        if self.pure_state_part is 1:
            data = 1
        else:
            data = make_multimode_H(self.pure_state_part, self.inds_pure_part, self.dims_state_list)
        return MyQobj(data, q_type = TYPE_STATE)
    
    def get_dens_part_super(self):
        self.dens_state_part_mqobj.to_super()
        return self.dens_state_part_mqobj
    
    def get_proj_to_pure_part(self, required_inds):
        
        state_out_list = []
        required_inds = sorted(list(required_inds))
        
        
        states_out_list = [ MyQobj(identity(self.dims_state_list[i])) 
                          if i not in self.inds_pure_part else self.state_mq_in_list[i]
                          for i in required_inds]
        
        proj = tensor_mq(states_out_list)
        proj.q_type = TYPE_OPERATOR
        return proj
        
    
    def complete_state_mq(self, state_in, state_inds, required_inds, dense = False):
        state_inds = list(state_inds)
        required_inds = sorted(list(required_inds))
        
        states_list = self.dense_states_mq_full_list if dense else self.state_mq_in_list
        inds_list = self.inds_list
        
        states_to_add_before = []
        states_to_add_after = []
        
        inds_before = []
        inds_after = []
        
        min_state_ind = min(state_inds) if len(state_inds)>0 else -1
        
        for ind in required_inds:
            if ind not in inds_before + state_inds + inds_after:
                for kk, inds in enumerate(inds_list):
                    if ind in inds:
                        
                        if ind < min_state_ind:
                            states_to_add_before += [states_list[kk]]
                            inds_before += inds
                        else:
                            states_to_add_after += [states_list[kk]]
                            inds_after += inds
                            
#                             states_to_add += [states_list[kk]]
#                             state_inds += inds
        
        states_in_list = [] if state_in is None else [state_in]
        states_out_list = states_to_add_before + states_in_list + states_to_add_after
        state_inds_out = inds_before + state_inds + inds_after
        
        return tensor_mq(states_out_list), state_inds_out
    
    def complete_state(self, state_in, state_inds, required_inds, dense = False):
        state_inds = list(state_inds)
        required_inds = list(required_inds)
        
        states_list = self.dense_states_full_list
        inds_list = self.inds_list

        states_in_list = [] if state_in == None else [state_in]
        
        for ind in required_inds:
            if ind not in state_inds:
                for kk, inds in enumerate(inds_list):
                    if ind in inds:
                        if ind > max(state_inds):
                            states_in_list = states_in_list + [states_list[kk]]
                            state_inds = state_inds + inds
                        else:
                            states_in_list = [states_list[kk]] + states_in_list
                            state_inds = inds + state_inds
        
        if dense:
            states_out_list = [state if state.type is 'oper' else ket2dm(state) for state in states_in_list]
            
        return tensor(states_in_list), state_inds
        
        
#     def prepare_projectors(self, is_kraus):
#         if is_kraus:
            
#             # list of projectors for ancilla
#             if self.N_aux > 0:
#                 if self.N_aux_lvls > 0:
#                     raise NotImplementedError
                    
#                 self.aux_proj_list = get_aux_proj_list(self.I_sys_list, self.dim_aux_list, self.aux_inds, self.order)
                
#             elif self.N_aux_lvls > 0:
#                 self.aux_proj_list = get_aux_lvls_proj_list(self.N_aux_lvls, self.dim_sys_list)

#         else:
#             if self.N_aux_lvls > 0:
#                 raise NotImplementedError
                
#         self.N_aux_proj = len(self.aux_proj_list)
        
        
#     def plot(self, plot_size):
        
#         if len(self.coords_sys) ==0:
#             # all qubits in an array
            
#             coords_aux = self.aux_inds
#             for i in range(len(coords_aux)):
#                 if coords_aux[i]<0: coords_aux[i] = coords_aux[i]+self.N_modes
                    
#             coords_sys = np.setdiff1d(np.arange(self.N_modes),coords_aux)
            
#             coords_aux = [[coord,0] for coord in coords_aux ]
#             coords_sys = [[coord,0] for coord in coords_sys ]
#         else:
#             coords_sys = self.coords_sys
#             coords_aux = self.coords_aux
            
#         plot_scheme(coords_sys, coords_aux, plot_size)
TAG_GHZ = 'GHZ'
TAG_SCHWING = 'SCHWINGER'
TAG_SCHWING_H_INV = 'SCHWINGER HALF INVERSE'
TAG_ISING = 'ISING'
TAG_VAC = 'VACUUM'
TAG_EX = 'EXITED'
TAG_PLUS = 'PLUS'
TAG_MINUS = 'MINUS'
TAG_CL_ISING = 'Cluster Ising'
TAG_HALD_RYD = 'Haldane Model with Ryd.'
TAG_HALD_IONS = 'Haldane Model with ions'
TAG_GHZ_ANTIFERR = 'GHZ anti-ferromag'
TAG_SSH_P = 'SSH model plus'
TAG_SSH_M = 'SSH model minus'
TAG_SSH_P_4 = 'SSH model plus, 4 gr. states'
TAG_W = 'Parrent H for W state'
TAG_MG = 'H of Majumdar-Gosh model'
TAG_CLUST = 'Cluster state'
TAG_SQUEEZED = 'Squeezed state'
TAG_AKLT = 'AKLT model'

class Model(object):
    '''
    Describes the target model
    
    Parameters
    ----------
    TAG : str
        One of the tags existing tags.
    N_sys : int
        Size of the target model
    system : System object or None
        Required for some some models.
    C_ops : list of jump operators acting to local qubits, 
        which are converted to multiple operators for each (set) of qubits.
        If are given, optimize steady state
    '''

    def __init__( self, TAG, N_sys, system = None, psi_targ_q = None, proj = None, 
                 args = None, H_cost_cust = None, C_ops = [], steady_st_cost = [], H_cost = None):
        self.TAG = TAG
        self.args = args
        self.N_sys = N_sys
        N_gr_states = 1
        
        if psi_targ_q is not None:
            self.psi_targ_q = psi_targ_q
            self.psi_targ = psi_targ_q.full()
            self.psi_targ_list = [psi_targ_q]
            H_cost = -psi_targ_q * psi_targ_q.dag()
            self.H_cost = H_cost
        elif H_cost is not None:
            self.H_cost = H_cost
        else:
            if TAG == TAG_SCHWING:
                J, w, m, e0 = 1, 1, -0.5, 0
                if args is not None:
                    J, w, m, e0 = args
                H_cost = get_schw_H_revers(N_sys, J, w, m, e0)
#                 H_cost = get_schw_H(N_sys, J, w, m, e0)
            if TAG == TAG_SCHWING_H_INV:
                H_cost = get_schw_H_half_invert(N_sys)
            elif TAG == TAG_ISING:
                H_cost = get_ising_H(N_sys)
            elif TAG == TAG_CL_ISING:
                H_cost = get_cl_ising_H(N_sys, 1, 1)
            elif TAG == TAG_GHZ:
                H_cost = get_GHZ_H(N_sys)
            elif TAG == TAG_GHZ_ANTIFERR:
                H_cost = get_GHZ_antiferromag_H(N_sys)
            elif TAG == TAG_HALD_RYD:
                H_cost = get_H_Hald_qubits(N_sys, trans_map_ryd)
            elif TAG == TAG_HALD_IONS:
#                 H_cost = get_H_Hald_qubits(N_sys, trans_map_ion)
                J, Delt, D2, l = 1, 1, 1, 0
                if args is not None:
                    J, Delt, D2, l = args
                H_cost = get_H_Hald_qub(N_sys, J, Delt, D2, l)
            elif TAG == TAG_EX:
                H_cost = sum(make_multimode_oper(sigmaz(), N_sys, [i]) for i in range(N_sys))

            elif TAG == TAG_VAC:
                H_cost = -sum(make_multimode_oper(sigmaz(), N_sys, [i]) for i in range(N_sys))
            elif TAG == TAG_PLUS:
                H_cost = -sum(make_multimode_oper(sigmax(), N_sys, [i]) for i in range(N_sys))
            elif TAG == TAG_MINUS:
                H_cost = sum(make_multimode_oper(sigmax(), N_sys, [i]) for i in range(N_sys))
            elif TAG == TAG_SSH_M:
                H_cost = get_SSH_H(N_sys, t_p=-1, t_m=-2)
            elif TAG == TAG_SSH_P_4:
                H_cost = get_SSH_H(N_sys, t_p=-1, t_m=2)
                N_gr_states = 4
            elif TAG == TAG_SSH_P:
#                 H_cost = get_SSH_H(N_sys, T_p, T_m, b)
                t_p, t_m, B = 1, 0, 0.1
                if args is not None:
                    t_p, t_m, B = args
                H_cost = get_SSH_H(N_sys, t_p=t_p, t_m=t_m, B = B)
            elif TAG == TAG_W:
                H_cost = get_W_H(N_sys)
                
            elif TAG == TAG_MG:
                H_cost = get_MG_H(N_sys)
            elif TAG == TAG_SQUEEZED:
                H_cost = get_squeez_H(N_sys)
            elif TAG == TAG_CLUST:
                b = 0
                if args is not None:
                    b = args[0]
                H_cost = get_clust_H(N_sys, b)
            elif TAG == TAG_AKLT:
                l, b = 0, 0
                if args is not None:
                    l, b = args
                H_cost = get_AKLT_H_2_2(N_sys, l, b)
            
            elif H_cost_cust is not None:
                H_cost = H_cost_cust


            # delete blockaded dimensions
            if system is not None and isinstance(system, SystemRyd) and system.block_threshold != np.inf:
                dims_to_del = get_dims_to_del(system.coords_sys, system.block_threshold)
                H_cost = map_ryd(H_cost.data, dims_to_del)
#             H_cost = H_cost*H_cost
            if not isinstance(H_cost, Qobj):
                H_cost = Qobj(H_cost)
    #         H_cost = H_targ_rand
            self.H_cost = H_cost

        vals, vecs = H_cost.eigenstates()
        self.vals, self.vecs = vals, vecs

        if TAG == TAG_SCHWING:
            self.psi_targ_q = find_zero_magnet_state(vecs)
            self.psi_targ_list = [self.psi_targ_q]
        else:
            self.psi_targ_q = vecs[0]
            self.psi_targ_list = [vecs[i] for i in range(N_gr_states)]
        
        if proj is not None:
            self.H_cost = proj * H_cost * proj.dag()
            self.psi_targ_q = proj * self.psi_targ_q
            self.psi_targ_list = [proj * psi for psi in self.psi_targ_list]
            
        self.psi_targ = self.psi_targ_q.full()
        self.psi_targ_dag = self.psi_targ_q.dag().full()
        
        self.E_max = vals[-1]
        self.E_min = vals[0]

        # U_H converts H_cost to diagonal
        basis_stack = sp.sparse.csr_matrix(sp.sparse.hstack([b.data for b in vecs]))
        self.U_H = Qobj(
            fastsparse.fast_csr_matrix([basis_stack.data, basis_stack.indices, basis_stack.indptr]),
            dims = H_cost.dims
        )
        if len(C_ops) != 0:
            C_ops_full = []
            modes_dims_list = H_cost.dims[0]
            for C in C_ops:
                n_c = get_q_N_mode(C)
                for i in range(N_sys-n_c+1):
                    inds_to_act = list(range(i, i+n_c))
                    C_ops_full += [make_multimode_H(C, inds_to_act, modes_dims_list)]
             
            steady_st_cost_full = []
            for st in steady_st_cost:
                n_c = get_q_N_mode(st)
                for i in range(N_sys-n_c+1):
                    inds_to_act = list(range(i, i+n_c))
                    steady_st_cost_full += [make_multimode_H(st, inds_to_act, modes_dims_list)]
            
            
            self.C_ops_full = C_ops_full
            self.steady_st_cost_full = steady_st_cost_full
        
    def get_energy(self, state):
        # trace(H_cost * state)
        
        if state.data.shape[0] != self.H_cost.shape[0]:
            raise ValueError("shape of state and psi_targ_q is incompatible")
            
        if isinstance(state, Qobj) or isinstance(state, MyQobj):
            state = state.data
           
        if state.shape[1] == 1:
            res = state.conj().T * (self.H_cost.data * state)
        else:
            res = self.H_cost.data * state
        return np.real(sum(res.diagonal()))
        
    def get_fidelity(self, state):
        # trace(state_targ * state)
        
        if state.data.shape[0] != self.psi_targ_q.shape[0]:
            raise ValueError("shape of state and psi_targ_q is incompatible")
            
        if isinstance(state, Qobj) or isinstance(state, MyQobj):
            state = state.data
        
        return sum(
            np.real(sum((self.psi_targ_q.dag().data * state * psi_t.data).diagonal()))
            for psi_t in self.psi_targ_list
        )
    
    def get_st_st_cost(self, state):
        
        state.dims = [state.dims[1], state.dims[1]]
        if not isinstance(state, Qobj):
            state = state.to_qobj()
        return sum(abs(f(state, self.H_cost, self.C_ops_full, k)) 
                   for k in self.steady_st_cost_full)
        
        
def f(rho, H, c_list, k):
    
    h = -1j*(k*H-H*k) * rho
    
    L1 = sum(c.dag() * k * c for c in c_list ) * rho
    L2 = -0.5*sum(k * c.dag()*c for c in c_list ) * rho
    L3 = -0.5*sum(c.dag()*c*k for c in c_list ) * rho
    
    return (h+L1+L2+L3).tr()/2
def find_zero_magnet_state(vecs):
    N_modes = get_q_N_mode(vecs[0])
    H_z_sum = sum(make_multimode_oper(S_z, N_modes, [i]) for i in range(N_modes))
    
    for v in vecs:
        if abs((v.dag()*H_z_sum*v).tr()) < 1e-8:
            return v
    
    raise Exception('No 0 magnetization state')
def is_zero_magnet(v):
    N_modes = get_q_N_mode(v)
    H_z_sum = sum(make_multimode_oper(S_z, N_modes, [i]) for i in range(N_modes))
    
    if abs((v.dag()*H_z_sum*v).tr()) < 1e-8:
        return True
    return False
def filt_zero_magnet_state(vecs):
    N_modes = get_q_N_mode(vecs[0])
    H_z_sum = sum(make_multimode_oper(S_z, N_modes, [i]) for i in range(N_modes))
    
    zero_list = []
    for v in vecs:
        if abs((v.dag()*H_z_sum*v).tr()) < 1e-8:
            zero_list += [v]
    
    return zero_list
class Gate(object):
    
    is_dissip = False
    
    def __init__( self, H_q_list, inds_to_act_list = None, fixed_params = [], bounds = [(-np.inf, np.inf)], lind_diss = 0, to_diag = True):
        
        '''
        Parameters
        ----------
        H_q_list : list of Qobj
            Hamiltonians of the gate. Each of them has a corresponding free parameter t_i, 
            such that the resultet Hamiltonian would be H(t_0, t_1, ...) = t_0 * H_0 + t_1 * H_1 + ...
        system : System object
        inds_to_act : array-like
            Indices of the aux+sys state, to which the gate is applyed. If [], is applyed to all.
        fixed_params : array-like
            If [], no fixed params. If [None, t_0_fixed, t_1_fixed, ...] fixes not None params for H_q_list in the corresponding order.
        bounds : [(-min_0, max_0),(-min_1, max_1),...]
            Bounds for parameters of H_q_list in the corresponding order. If only one bound is given, 
            is used for all params.
        lind_diss : Qobj
            Lindblad dissipator. The last of gates parameters corresponds to time during which 
            lind_diss acts to the state.
        name: str
            Name for the gate to be shown.
        '''
        
        if not isinstance(H_q_list, list):
            H_q_list = [H_q_list]
            if inds_to_act_list is not None:
                inds_to_act_list = [inds_to_act_list]
        
        if inds_to_act_list is None:
            inds_to_act_list = [list(range(get_q_N_mode(H))) for H in H_q_list]
            
        self.H_q_list = H_q_list
        self.N_H = len(H_q_list)
        self.N_params = self.N_H
        self.inds_to_act_list = inds_to_act_list
        
        if not isinstance(fixed_params, collections.Iterable):
            fixed_params = [fixed_params]
        self.set_fixed_params(fixed_params)
        
        if lind_diss != 0:
            self.is_dissip = True
            lind_diss = MyQobj(lind_diss, TYPE_HAMILTONIAN, is_super = True)
        self.lind_diss = lind_diss
        
        bounds = list(bounds)
        if len(bounds) == 1:
            bounds = bounds * self.N_params
        self.bounds = bounds
        
        self.dims_list = [get_q_dim_list(H) for H in H_q_list]
        
        self.to_diag = to_diag
        
    def set_fixed_params(self, fixed_params):
            
        if len(fixed_params) == 0:
            fixed_params = np.array([None]  * self.N_H)
        elif len(fixed_params) != self.N_H:
            raise ValueError(
                "Number of fixed_params has to be equal to the number of Hamiltonians with None for non fixed purams")
            
        self.fixed_params = np.array(fixed_params)
        self.where_not_fixed = np.where(self.fixed_params == None)[0]

        self.N_free_params = len( [p for p in fixed_params if p == None]  )
        
        
    def prepare(self, modes_dims_list, do_sparse):
        '''
        Complete the space of the Hamiltonians to act to modes_dims_list
        '''
        
        
        if (self.is_dissip or self.N_H != 1):
            H_list = [make_multimode_H(self.H_q_list[ii], self.inds_to_act_list[ii], modes_dims_list) 
                          for ii in range(self.N_H)]
        else:
            H_list = self.H_q_list
        
        
        U = 1
        U_data = 1
        if self.is_dissip:
            
            self.H_list = [MyQobj(-1j*(spre(H)-spost(H)), TYPE_HAMILTONIAN, is_super = True) for H in H_list]
            
        elif self.N_H == 1:
            
            H = H_list[0]
            inds_to_act = self.inds_to_act_list[0]
            
            if self.to_diag and is_diag(H):
                H_diag = H
            else:
                _, basis = H.eigenstates()
                U = Qobj(sp.sparse.hstack([b.data for b in basis]), dims = H.dims)
                H_diag = U.dag() * H * U
                
            if len(inds_to_act) != 0:
                H_diag = make_multimode_H(H_diag, inds_to_act, modes_dims_list)
                U = make_multimode_H(U, inds_to_act, modes_dims_list)
            
            if U != 1:
                if do_sparse:
                    U_data = U.data
                else:
                    U_data = U.full()
            else:
                U_data = 1

            self.H_list = [MyQobj(-1j*H_diag.diag(), TYPE_HAMILTONIAN, dims = H_diag.dims)]
        
        elif do_sparse:
            
#             if norm(np.imag(sum(H_list).data.data)) == 0:
#                 # make sparse Hamiltonians real
#                 with warnings.catch_warnings():
#                     warnings.simplefilter("ignore")
#                     H_data_list = [sp.sparse.csr_matrix(H.data, 
#                                                         dtype = np.float64) for H in self.H_q_list]
    
            self.H_list = [MyQobj(-1j*H, TYPE_HAMILTONIAN) for H in H_list]
                
        else:
            
            self.H_list = [MyQobj(-1j*H.full(), TYPE_HAMILTONIAN, dims = H.dims) for H in H_list]
    
        self.U = MyQobj(U_data, TYPE_OPERATOR, dims = U.dims if U != 1 else None)
    
    
    def get_H(self, params):
        
        if len(params) != self.N_free_params:
            raise ValueError("len(params) != self.N_free_params")
        
        where_not_fixed = self.where_not_fixed
        
        params_to_set = copy(self.fixed_params)
        params_to_set[where_not_fixed] = params
        params_to_set = cut_params(params_to_set, self.bounds)
        H_out = sum( self.H_list[i]*params_to_set[i] for i in range(self.N_params) ) 

        t = abs(params_to_set[0])
#         t = max(np.abs(params_to_set))
        H_out += t * self.lind_diss
        
        return H_out
    
    def apply(self, params, A, A_var_list, jac, params_var_list):

        H = self.get_H(params)

        A_out = H.expm_multiply(A)
        
        if jac:
            
            # Apply variation of the Hamiltonian 
            for ind, params_var in params_var_list:
                H_var = self.get_H(params_var)
                A_var = A_var_list[ind]
                # If A_var is 0, H_var is applied to A since it means that it was the same so far
                A_var_list[ind] = H_var.expm_multiply(A) if A_var is 0 else H_var.expm_multiply(A_var)

            # Proceed the rest of variation of the evolution
            params_var_inds = [p[0] for p in params_var_list]
            for kk, A_var in enumerate(A_var_list):
                if kk not in params_var_inds and A_var is not 0:
                    A_var_list[kk] = H.expm_multiply(A_var)
        
        return A_out, A_var_list
def make_multimode_H(H, inds_to_act, modes_dims_list):

    if H is 1:
        return H
    elif not isinstance(H, Qobj):
        raise ValueError('H has to be Qobj or 1')
    
    if isinstance(inds_to_act, (int, np.int64, float, complex)):
        inds_to_act = [inds_to_act]
    inds_to_act = np.array(inds_to_act)
    modes_dims_list = np.array(modes_dims_list)
    dims_check = modes_dims_list[inds_to_act]

    if list(dims_check) != H.dims[0]:
        raise ValueError('Incompatible dimensions')

    others_dims = list(modes_dims_list)
    for i in sorted(inds_to_act)[::-1]: del others_dims[i]

    H_out_list = [H] + [identity(dim) for dim in others_dims]
    
    N_modes = len(modes_dims_list)
    N_H_modes = len(inds_to_act)
    
    order = np.array([None]*N_modes)
    order[inds_to_act] = range(N_H_modes)
    order[np.where(order == None)] = range(N_H_modes, N_modes)

    return my_permute(tensor(H_out_list), order)
I_spar_mq_cash = {}
I_dens_mq_cash = {}
def make_multimode_H_mq(H, inds_to_act, modes_dims_list):

    if H is 1:
        return H
    elif not isinstance(H, MyQobj):
        raise ValueError('H has to be MyQobj or 1')
    
    global I_spar_mq_cash, I_mq_dens_cash, I_cash
    I_cash = I_dens_mq_cash if H.format is FORMAT_DENSE else I_spar_mq_cash
    
    if isinstance(inds_to_act, (int, np.int64, float, complex)):
        inds_to_act = [inds_to_act]
    inds_to_act = np.array(inds_to_act)
    modes_dims_list = np.array(modes_dims_list)
    
    dims_check = modes_dims_list[inds_to_act]

    H_dims = H.dims if isinstance(H, Qobj) else H.dims

    if list(dims_check) != H_dims[0] and list(dims_check) != H_dims[1]:
        raise ValueError('Incompatible dimensions')

    others_dims = list(modes_dims_list)
    for i in sorted(inds_to_act)[::-1]: del others_dims[i]
    
    for dim in others_dims:
        if dim not in I_cash:
            I = identity(dim).full() if H.format is FORMAT_DENSE else identity(dim)
            I_cash[dim] = MyQobj(I)

    H_out_list = [H] + [I_cash[dim] for dim in others_dims]
    
    N_modes = len(modes_dims_list)
    N_H_modes = len(inds_to_act)
    
    order = np.array([None]*N_modes)
    order[inds_to_act] = range(N_H_modes)
    order[np.where(order == None)] = range(N_H_modes, N_modes)
    H_out = tensor_mq(H_out_list)

    return my_permute_mq(H_out, order)
def my_lindblad_dissipator(c_with_inds_list, modes_dims_list):
    return sum( 
        lindblad_dissipator( make_multimode_H(c, inds, modes_dims_list) )
        for c, inds in c_with_inds_list
    )
class ParamsMap(object):
    
    def __init__( self, map_list = [] ):
        '''
        max_params - maximum number of parameters
        map_list is [[ind_to_map, inds_main, func],...] 
        inds_main is a list of indises
        such that param[ind_to_map] = func[inds_main]
        '''
        
        self.map_list = sorted(map_list, key = lambda x: x[0])
        self.N_maps = len(map_list)
        
        self.to_map_inds = [m[0] for m in map_list]
    
    def set_params(self, params ):
        self.params = params
        max_params = len(params) + len(self.to_map_inds)
        self.main_inds = [i for i in range(max_params) if i not in self.to_map_inds]
        
        params_new = [None]*max_params
        # d_params_new consists of list of tuples with indice the parameter depends on and the param+increment
        d_params_new = [None]*max_params

        for jj, ind in enumerate(self.main_inds):
            params_new[ind] = params[jj]
            d_params_new[ind] = [(ind, params[jj] + eps)]
        
        for m in self.map_list:
            ind_to_map = m[0]
            main_inds = m[1]
            func = m[2]
            
            main_params = [params_new[j] for j in main_inds]
            if None in main_params:
                raise ValueError('None in main_params, need to imrove the code')
            # calculate parameter    
            params_new[ind_to_map] = func( main_params )
            # calculate increments
            d_params_map = []
            for jj, ind in enumerate(main_inds):
                d_main_params = copy(main_params)
                d_main_params[jj] = d_params_new[ind][0][1]
                
                d_params_map += [(ind,func( d_main_params ))]
                
            d_params_new[ind_to_map] = d_params_map
        
        self.params_new = params_new
        self.d_params_new = d_params_new
        
    # return parameters for a gate and all sets of increments of 
    # variations of parameters with indeses of variated parameter
    def pick_params(self, inds):
        try:
            params_out = [self.params_new[i] for i in inds]
        except:
            print('     ',len(self.params_new), inds)
        d_params_out_list = []
        
        for jj, ind in enumerate(inds):
            d_params_ind = self.d_params_new[ind]
            for d in d_params_ind:
                d_params_out = copy(params_out)
                d_params_out[jj] = d[1]
                d_params_out_list += [(d[0],d_params_out)]
        return params_out, d_params_out_list
                
    
    
    def map_params(self, list_in):
        list_out = list(copy(list_in))
        for dep in self.dep_list:
            list_out.insert(dep[0], list_out[dep[1]] * dep[2] + dep[3])

        return np.array(list_out)
    
    def reverse_gradient_map(self, list_in):
        list_out = copy(list_in)
        for dep in self.dep_list[::-1]:
            list_out[dep[1]] += list_out[dep[0]] * dep[2]
            del list_out[dep[0]]
        
        return list_out
class GatesSequence(object):
    '''
    Object for a gate sequence in 'gates_list'. 
    
    Parameters
    ----------
    gates_list : list of objects Gate
    system : System object
    '''
    def __init__( self, gates_list, params_map = ParamsMap([]) ):
        
        self.gates_list = [deepcopy(g) for g in gates_list]
        self.params_map = params_map
        self.N_gates = len(self.gates_list)
        
        self.count_free_params()
        
        self.init_space_to_act(self.gates_list)
        
    def set_fixed_params(self, params):
        if isinstance(params, np.ndarray):
            params = list(params)
            
        if not isinstance(params, list):
            params = [params]*self.N_params
            
        elif len(params) != self.N_params:
            raise Exception('len(params) != self.N_params')
        
        ind = 0
        for gate in self.gates_list:
            params_gate_fixed, ind = pick_params(params, ind, gate.N_params)
            gate.set_fixed_params(params_gate_fixed)

        self.count_free_params()
        
    def unfix_params(self):
        for gate in self.gates_list:
            gate.set_fixed_params([])
            
        self.count_free_params()
    
    def count_free_params(self):
        # Calculate N_params in GateSequence
        self.N_params = sum( gate.N_params for gate in self.gates_list)
        N_free_params = sum( gate.N_free_params for gate in self.gates_list)
        # Number of free parameters
        self.N_free_params = N_free_params - self.params_map.N_maps
        
        fixed_params = []
        for gate in self.gates_list:
            fixed_params = np.append(fixed_params, gate.fixed_params)
        self.fixed_params = fixed_params
        
    def init_space_to_act(self, gates_list):
        '''
        Prepare a minimal system space to act.
        '''
        inds_to_act_unsort = []
        dims_to_act_unsort = []
        for g in gates_list:
            for i in range(g.N_H):
                inds_to_act_unsort += g.inds_to_act_list[i]
                dims_to_act_unsort += g.dims_list[i]
                
        inds_to_act_unsort = np.array(inds_to_act_unsort)
        dims_to_act_unsort = np.array(dims_to_act_unsort)

        inds_to_act = np.array(list(set(inds_to_act_unsort)))
        dims_to_act = []
        
        # Check if dims corresponded to one index in different gates are the same
        for ind in inds_to_act:
            dims_list = dims_to_act_unsort[np.where(inds_to_act_unsort == ind)]
            dim = dims_list[0]
            if np.count_nonzero(dims_list == dim) != len(dims_list):
                raise Exception('Dimension mismatch in the gates list')
            dims_to_act += [dim]
        
#         # Check if the indices are serial
#         if max(inds_to_act) != len(inds_to_act)-1:
#             raise NotImplementedError('Indices of the gate sequence FOR NOW have to be serial')
            
        self.inds_to_act = inds_to_act
        self.dims_to_act = dims_to_act
        
    def prepare(self, st_in = 1, do_sparse = True, dims_to_act = None):
        
        if dims_to_act is None:
            dims_to_act = self.dims_to_act
            
        for gate in self.gates_list:
            gate.prepare(dims_to_act, do_sparse)
            
        # Get intermediate unitaries
        self.interm_U_list = self.get_interm_U_list(self.gates_list)

        self.interm_U_list[0] *= st_in
        self.interm_U_list[0].to_dense()
        self.interm_U_list[0].q_type = TYPE_OPERATOR
    
    def get_interm_U_list(self, gates_list):
        
        U_list = [gate.U for gate in gates_list]
#         U_list_dag = [ U.dag() for U in U_list ]
        
        # List of intermediate matrix for trasition between bases in which different gates are diagonal if they are diagonalized. The evolution then is
        # ket_out = ... * interm_U_list[1] * U_exp(gate_1) * interm_U_list[0] * ket_in
        interm_U_list = [ np.dot(U_list[i+1].dag(), U_list[i]) for i in range(len(U_list) - 1) ]
        # The first and the last U
        interm_U_list = [U_list[0].dag()] + interm_U_list + [U_list[-1]]
        
        return interm_U_list
    
    
    def apply(self, params, jac, save = False):
        
        params_map = self.params_map
        params_map.set_params(params)
        
        gates_list = self.gates_list
        interm_U_list = self.interm_U_list 

        A = interm_U_list[0]
        # Variation of the evolution
        A_var_list = [0]*self.N_params
        ind = 0
        for i, gate in enumerate(gates_list):
            params, params_var = params_map.pick_params(range(ind, ind+gate.N_free_params))
            ind += gate.N_free_params
            A, A_var_list = gate.apply(params, A, A_var_list, jac, params_var)
            
            # Apply unitary which change bases in which the Hamiltonians are diagonal
            interm_U_list[i+1].to_dense()
            A = interm_U_list[i+1]*A
            A_var_list = [0 if A_var is 0 else interm_U_list[i+1]*A_var for A_var in A_var_list]
            
        A_var_list = [A_var for A_var in A_var_list if A_var != 0]
            
        return A, A_var_list
    
    def gen_unitary(self, params, jac):
        self.A, self.A_var_list = self.apply(params, jac)
        self.A.to_sparse()
        [A.to_sparse() for A in self.A_var_list]
        
        self.A_dag = self.A.dag()
        self.A_dag_var_list = [A.dag() for A in self.A_var_list]
#         if A.format not in [FORMAT_DENSE, FORMAT_SPARSE]:
#             raise NotImplementedError
            
#         dims = [self.dims_to_act]*2
#         self.A = Qobj(A.data, dims = dims)
#         self.A_var_list = [Qobj(A_var.data, dims = dims) for A_var in A_var_list]
    
    def gen_unitary_qobj(self, params, jac):
        A, A_var_list = self.apply(params, jac)
        if A.format not in [FORMAT_DENSE, FORMAT_SPARSE]:
            raise NotImplementedError
            
        dims = [self.dims_to_act]*2
        self.A = Qobj(A.data, dims = dims)
        self.A_var_list = [Qobj(A_var.data, dims = dims) for A_var in A_var_list]
    
    def gen_unitary_dens(self, params, jac):
        self.last_params = params
        self.A, self.A_var_list = self.apply(params, jac)        
        self.A_dag = self.A.dag()
        self.A_dag_var_list = [A.dag() for A in self.A_var_list]
class KrausGenerator(object):
    '''
    Object for generation of Krauses.
    
    Parameters
    ----------
    gates_list : list of objects Gate
    system : System object
    opt_props : Optimiz_props object
    gates_BA_list : list of Gate lists which are applied depended on the ancillas outcome. 
        Number of lists should be equal to the number of ancillas outcome
    params_map: parameters dependenences
    '''
    
    aux_proj_list = []
    K_list = []
    def __init__( self, gates_list, opt_props, gates_BA_list = [], params_map = ParamsMap([])):

        self.gates_list = gates_list
        self.N_gates = len(gates_list)
        self.gates_BA_list = gates_BA_list
        self.opt_props = opt_props
    
    def prepare(self, system):
        
        self.gates_seq = GatesSequence(self.gates_list)
#         self.gates_seq.prepare(do_sparse = self.opt_props.do_sparse, system = system)
        
        st_in = system.get_proj_to_pure_part(self.gates_seq.inds_to_act)
        
        self.gates_seq.prepare(st_in = st_in)
#         self.gates_seq.prepare(st_in = system.get_pure_part_in_full_space(), dims_to_act = system.dims_state_list)
        
        self.modes_dims_list = system.dims_state_list
        self.aux_proj_list = get_aux_proj_list(system, self.gates_seq.inds_to_act)
        self.N_aux_proj = len(self.aux_proj_list)
        
        if len(self.gates_BA_list) == 0:
            self.gates_BA_list = [None] * self.N_aux_proj
        elif len(self.gates_BA_list) != self.N_aux_proj:
            ValueError('len(gates_BA_list) != system.N_aux_proj')
            
        # projectors with back action
        self.aux_proj_list = [
            Projector(self.aux_proj_list[i], self.gates_seq.interm_U_list[-1], 
                      self.gates_BA_list[i]) 
            for i in range(self.N_aux_proj)
        ]

        self.gates_seq.interm_U_list[-1] = MyQobj(1, TYPE_OPERATOR)
        
        # Total params: params of gates before measurement + BA gates
        self.N_free_params = self.gates_seq.N_free_params + sum( p.N_free_params for p in self.aux_proj_list)
        self.N_params = self.gates_seq.N_params + sum( p.N_params for p in self.aux_proj_list)
    
    def extend_dims(self, inds_to_act, system):
            
            modes_dims_list = [system.dims_state_list[inds[0]] for inds in system.logic_mode_inds]
            
            self.K_list = [ make_multimode_H_mq(K, inds_to_act, modes_dims_list) 
                          for K in self.K_list]
            
            self.K_dag_list = [K.dag() for K in self.K_list]
            
            for i, K_var_list in enumerate(self.K_var_lists):
                self.K_var_lists[i] = [ make_multimode_H_mq(K, inds_to_act, modes_dims_list) 
                          for K in K_var_list]
        
    def gen_krauses(self, params):
        '''
        return a list of Kraus operators corresponding to the given 'params'
        '''
        jac = self.opt_props.jac
        
        self.K_list = []
        # List of lists of differentials of karauses [[d_0 E_0, d_1 E_0, ...],[d_0 E_1, d_1 E_1, ...],...]
        self.K_var_lists = []
        
        ind = self.gates_seq.N_free_params
        A, A_var_list = self.gates_seq.apply( params, jac)

#         dA_list = [(A_var + -1 * A)*(1/eps) for A_var in A_var_list]
        
#         A = make_multimode_H(A.to_qobj(), self.gates_seq.inds_to_act, self.modes_dims_list)
#         dA_list = [make_multimode_H(dA.to_qobj(), self.gates_seq.inds_to_act, self.modes_dims_list)
#                   for dA in dA_list]
        
        for nn, p in enumerate(self.aux_proj_list):
            params_BA, ind = pick_params(params, ind, p.N_free_params)
            
            dK_lists_add = []
            if p.gates_seq!=None:
                U, U_var_list = p.gates_seq.apply( params_BA, jac)
                K = p.proj * A
                K = U*K
                
#                 raise NotImplementedError
#                 U_1, dU_1_list = self.set_params_to_gates(p.gate_seq, params_BA, opt_props)
                
#                 if opt_props.jac:
#                     proj_1 = np.dot(U_1, proj)
#                     K = np.dot(proj_1, U_0)
#                     dK_lists_add = [ np.dot(proj_1, dU) for dU in dU_list ]
                    
                    
#                     K_1 = np.dot(proj, U_0)
#                     for kk, p2 in enumerate(self.proj_list):
#                         if kk == nn:
#                             dK_lists_add += [ np.dot(dU_1, K_1) for dU_1 in dU_1_list ]
#                         else:
#                             dK_lists_add += [ 0 ] * p2.N_free_params
#                 else:
#                     K = np.dot(proj, U_0)
#                     K = np.dot(U_1, K)
            else:
                K = p.proj * A

                if self.opt_props.jac:
                    K_var_lists_add = [ p.proj * A_var for A_var in A_var_list ]
            
            if self.opt_props.jac:
                self.K_var_lists += [K_var_lists_add]
                
            self.K_list += [ K ]
            
        if self.K_list[0].is_super:
#             raise NotImplementedError
            self.map_sup = sum(self.K_list)
            if self.opt_props.jac:
                self.map_sup_var_list = [sum(K_var_list[i] for K_var_list in self.K_var_lists) 
                                     for i in range(len(self.K_var_lists[0]))]
        else:
            self.K_dag_list = [K.dag() for K in self.K_list]
        
    def get_sup_map(self):
        if len(self.K_list) == 0:
            raise ValueError
        return sum(k.to_super(is_copy = True) for k in self.K_list)
class Projector(object):
    '''
    Projector describing measurement
    '''
    gates_seq = None
    N_free_params = 0
    N_params = 0
    def __init__( self, proj, U_last, gates_list = None):
        
        proj = MyQobj(proj, TYPE_OPERATOR)
        
        self.proj = proj * U_last
        
        if gates_list != None:
            self.set_gates_seq( GatesSequence( gates_list ) )
            self.gates_seq.interm_U_list[0].to_dense()
            self.proj = self.gates_seq.interm_U_list[0] * self.proj
            self.gates_seq.interm_U_list[0] = MyQobj(1, TYPE_OPERATOR)
        
#         proj_interm_dag = proj_interm.T.conj()
        
#         self.proj_interm = proj_interm
#         self.proj_interm_dag = proj_interm_dag
        
        
    def set_gates_seq(self, gates_seq):
        
        gates_seq.prepare()
        self.gates_seq = gates_seq
        
        # Parameters of all gates
        self.N_free_params = gates_seq.N_free_params
        self.N_params = gates_seq.N_params
        
        N_layers = gates_seq.N_gates
        self.N_layers = N_layers
class Simulator(object):
    '''
    Object for simulation
    
    Parameters
    ----------
    system : System object
    opt_props : Optimiz_props object
    model : Model object. 
    params_init : array-like, [], or float. If [], the parameters are generated randomly. 
        If float, the parameters becomes the same float number.
    '''
    MIN_glob = np.inf
    def __init__( self, system, opt_props, model, params_init, params_map = None):

        self.system = system
        self.opt_props = opt_props
        self.params_init = params_init
        self.model = model
        self.params_map = params_map       
           
    def set_params_init(self, params_init):
        self.params_init = params_init
        params_map = self.params_map
        
        if params_map is not None:
                
            # New number of free parameters
            self.N_free_params = self.N_free_params - params_map.N_maps
            
        N_free_params = self.N_free_params
         
        # initialize initial parameters
        if isinstance(params_init, numbers.Number):
            self.params_init = np.repeat(params_init, N_free_params)
        elif len(params_init) == 0:
            np.random.seed(3)
            self.params_init = np.random.rand(N_free_params)* 2 * math.pi
#         elif len(params_init) != N_free_params:
#             sys.exit( "len(params_init) != N_free_params" )
    
    def optimize_params(self):
        res = optimize_params(self)
        return Result( self, res )
class SimulatorCoherent(Simulator):
    '''
    Object for simulation by applying gates to the state.
    
    Parameters
    ----------
    gates_seq : List of Gate objects which are applied in the corresponded order.
    N_layers : int or None.
        Do not partisipate in the simulation, but is shown in the print of results. 
        If None, -> total number of used gates.
    '''
    N_iter = 1
    def __init__( self, system, opt_props, model, gates_seq, params_init, N_layers = None, params_map = None):

        Simulator.__init__(self, system, opt_props, model, params_init, params_map)
        
        self.prepare(system, model)
        self.set_gates_seq(gates_seq, N_layers)
        self.set_params_init(params_init) 
        
    def prepare(self, system, model):
        
        # Convert the space of cost Hamiltonian and the target states-subspase to a larger space to simplify 
        # calculations of the energy and fidelity
        system.init_full_states()
        H_cost_with_aux = map_to_sub2full_space(model.H_cost, system.dims_state_list, system.logic_mode_inds)    
        self.H_cost_with_aux = MyQobj(H_cost_with_aux, TYPE_HAMILTONIAN)
        
        psi_targ_with_aux_list = [
            map_to_sub2full_space(psi_t, system.dims_state_list, system.logic_mode_inds)
            for psi_t in model.psi_targ_list
        ]
        self.psi_targ_with_aux_list = [
            MyQobj(psi_t, TYPE_HAMILTONIAN)
            for psi_t in psi_targ_with_aux_list
        ]
        
    def set_gates_seq(self, gates_seq, N_layers = None):
        gates_seq.prepare(st_in = self.system.get_pure_part_in_full_space(), dims_to_act = self.system.dims_state_list)
        self.gates_seq = gates_seq
        
        # Parameters of all gates
        self.N_free_params = gates_seq.N_free_params
        self.N_params = gates_seq.N_params
        
        if N_layers is None: 
            N_layers = gates_seq.N_gates
        self.N_layers = N_layers
    
    def simulation_test(self, param):
        return self.simulation([param]*self.N_params, return_F = True)
    
    
    def simulation_st_st(self, params, return_F = False):
        global params_glob
        params_glob = params
        if self.params_map is not None:
            params = self.params_map.map_params(params)
        
        N_p = self.gates_seq.N_free_params
        N = len(params)//N_p
        
        st_to_act = self.system.dens_state_part_mqobj
        
        rho_out = 0
        d_rho_out_list = [0]*N_p
        for i in range(N):
            
            params_loc = params[i*N_p:(i+1)*N_p]
            
            st_out, st_out_var_list = self.gates_seq.apply( params_loc, self.opt_props.jac)
            d_st_out_list = [(st_out_var + -1 * st_out)*(1/eps) for st_out_var in st_out_var_list]
        
            rho_out += st_out * st_to_act * st_out.dag()
            if self.opt_props.jac:
                d_rho_out_list = [d_rho_out_list[i]+ d_st_out_list[i] * st_to_act * d_st_out_list[i].dag() 
                                  for i in range(N_p)]
        
        rho_out = rho_out*(1/N)
        d_rho_out_list = [d/N for d in d_rho_out_list]
        
        en_cost = abs(model.get_st_st_cost(rho_out))
        grad_en_cost = []
        return SimResult(en_cost, 0, [], [], [], [],
                     rho_out, np.array(grad_en_cost))
            
    def simulation(self, params, return_F = False):

        if self.params_map is not None:
            params = self.params_map.map_params(params)

#         st_out, d_st_out_list = self.gates_seq.apply( params, self.opt_props.jac)
        st_out, st_out_var_list = self.gates_seq.apply( params, self.opt_props.jac)
        d_st_out_list = [(st_out_var + -1 * st_out)*(1/eps) for st_out_var in st_out_var_list]
        is_super = st_out.is_super
        if not is_super:
            st_to_act = self.system.dens_state_part_mqobj
            en_cost = self.get_energy_with_aux([st_out],[st_out], st_to_act)
            grad_en_cost = [
                self.get_energy_with_aux([st_out, d_st],[d_st, st_out], st_to_act) 
                for d_st in d_st_out_list
            ]
#             en_cost = -self.get_fidel_with_aux(st_out, st_to_act)
#             grad_en_cost = [
#                 -self.get_fidel_with_aux(d_st, st_to_act) 
#                 for d_st in d_st_out_list
#             ]
        else:
            st_to_act = self.system.get_dens_part_super()
            rho_out = st_out * st_to_act
            rho_out.vector_to_operator()
            
            d_st_out_list = [ d_st * st_to_act for d_st in d_st_out_list ]
            [ d_st.vector_to_operator() for d_st in d_st_out_list ]
            en_cost = self.get_energy_with_aux_dens(rho_out)
            grad_en_cost = [ self.get_energy_with_aux_dens(d_st) for d_st in d_st_out_list ]

        rho_out = 0
        if return_F:
            if is_super:
                st_to_act = self.system.get_dens_part_super()
                rho_out = st_out * st_to_act
                rho_out.vector_to_operator()
                F_cost = self.get_fidel_with_aux_dens(rho_out)
            else:
                st_to_act = self.system.dens_state_part_mqobj
                rho_out =  st_out * st_to_act * st_out.dag()
                F_cost = self.get_fidel_with_aux(st_out, st_to_act)
        else:
            F_cost = 0
        F_array = []
        E_array = []
            
        return SimResult(en_cost, F_cost, [], F_array, E_array, [],
                     rho_out, np.array(grad_en_cost))
        
    
    def get_energy_with_aux(self, L_list, R_list, state_to_act):
        
        # Here we permute action of evolution operator under trace and apply at first to the Hamiltonian
        A = sum( L_list[i].dag() *( self.H_cost_with_aux * R_list[i])
            for i in range(len(R_list))
        )
        # Here we apply to the rest of the state
        E = sum(( A * state_to_act).data.diagonal())
        
        return np.real(E)
    
    def get_fidel_with_aux(self, A, st_to_act):
        psi_targ_list = self.psi_targ_with_aux_list

        if st_to_act.data is 1:
            F = sum(
                abs((psi_targ.dag() * A).data[0][0])**2
                for psi_targ in psi_targ_list
            )
        else:
            F = sum(
                sum((psi_targ.dag() * A * st_to_act * A.dag() * psi_targ).data.diagonal())
                for psi_targ in psi_targ_list
            )
        
        return np.real(F)
    
    def get_energy_with_aux_dens(self, rho):

        return np.real(sum(( self.H_cost_with_aux * rho).data.diagonal()))
    
    def get_fidel_with_aux_dens(self, rho):
        
        psi_targ_list = self.psi_targ_with_aux_list
        
        F = sum(
                sum(( psi_targ.dag() * rho * psi_targ).data.diagonal())
                for psi_targ in psi_targ_list
            )
        
        return np.real(F)
class SimulatorKraus(Simulator):
    '''
    Object for simulation by preparing a Kraus map and applying it N_iter to the initial state.
    
    Parameters
    ----------
    kraus_gen_list : List of KrausGenerator objects which are applied in the corresponded order. 
        After each KrausGenerator the ancillas are resetted to the initial state.
    N_iter : number of iterations of application of kraus_gen_list before the cost function is measured.
    N_layers : int or None.
        Do not partisipate in the simulation, but is shown in the print of results. 
        If None, -> total number of used gates.
    '''
    
    def __init__( self, system, opt_props, model, kraus_gen_list, params_init, 
                 N_layers = None, params_map = None, N_iter = 10, krauses_order = None, measure_after_inds = None):

        
        self.N_iter = N_iter
        Simulator.__init__(self, system, opt_props, model, params_init, params_map)
        
#         self.system.prepare(opt_props.is_kraus)
        self.system.init_full_states()
        self.set_kraus_gen(kraus_gen_list, N_layers)
        self.set_params_init(params_init) 
        self.krauses_order = krauses_order
        self.measure_after_inds = measure_after_inds
        
    def set_kraus_gen(self, kraus_gen_list, N_layers = None):
        for kraus_gen in kraus_gen_list:
            kraus_gen.prepare(self.system)
            
        self.kraus_gen_list = kraus_gen_list
        
        # Parameters of all gates
        self.N_free_params = sum( k.N_free_params for k in kraus_gen_list )
        self.N_params = sum( k.N_params for k in kraus_gen_list )
        
        if N_layers is None: N_layers = [kraus_gen.N_gates for kraus_gen in kraus_gen_list]
        self.N_layers = N_layers
    
    def simulation_test(self, param):
        return self.simulation([param]*self.N_params, return_F = True)
    
    def simulation(self, params, return_F = False, mes_step = None, step_max = None):
        '''
        Return simulation results for given krauses 'krauses_list' and initial state 'rho_in' for steps 'step_max'.
        'ind_step' - index of the step which energy is minimized.
        'N_samp' - num of steps choosen for checking probabilities
        If 'state_targ' is given returns fidelities.
        '''
        
        opt_props = self.opt_props
        
        if isinstance(params, numbers.Number):
            params = [params]*self.N_free_params
#         params = map_params(params, self.params_map)
        if self.params_map is not None:
            params = self.params_map.map_params(params)
            
        # Henerate krauses for given parameters
        param_ind = 0
        for kraus_gen in self.kraus_gen_list:

            param_ind_new = param_ind + kraus_gen.N_free_params
            params_to_set = params[param_ind : param_ind_new]
            kraus_gen.gen_krauses(params_to_set)
            kraus_gen.param_ind_start = param_ind
            param_ind = param_ind_new
            
#             params_pick, ind = pick_params(params, ind, kraus_gen.N_free_params)
#             kraus_gen.gen_krauses(params_pick)
        
        krauses_order = self.krauses_order
        # prepare repeated krauses
        if krauses_order != None:
            kraus_gen_list = []
            for ind, inds_to_act in krauses_order:
                kraus_gen = deepcopy(self.kraus_gen_list[ind])
                kraus_gen.extend_dims(inds_to_act, self.system)
                kraus_gen_list += [kraus_gen]
        else:
            kraus_gen_list = self.kraus_gen_list
        
        if self.measure_after_inds is None:
            measure_after_inds = [len(kraus_gen_list)-1]
        else:
            measure_after_inds = self.measure_after_inds
        
        # Step where the energy is measured
        if mes_step == None:
            mes_step = self.N_iter

        # Max simulation step
        if step_max == None or step_max < mes_step:
            step_max = mes_step

        # probabilities
        prob_array_tot = []
        
                
        if self.kraus_gen_list[0].K_list[0].is_super:
            is_dissip = True
        else:
            is_dissip = False
            
        if is_dissip:
#             self.system.st_to_act.operator_to_vector()
#             rho_sup = self.system.st_to_act
            self.system.dens_state_part_mqobj.operator_to_vector()
            self.system.dens_state_part_mqobj.to_dense()
            rho_sup = self.system.dens_state_part_mqobj
        else:
#             rho = self.system.st_to_act
            rho = self.system.dens_state_part_mqobj
        rho_jac_list = [0]*self.N_free_params
        grad_en_cost = 0
        
        jac = opt_props.jac

        F_array, E_array, purity_array = [], [], []
        if return_F:
            if is_dissip:
                rho = rho_sup.vector_to_operator(is_copy = True)                    
            F_array += [self.model.get_fidelity(rho)]
            E_array += [self.model.get_energy(rho)]
            purity_array += [purity(rho)]
        E_cost = 0
        F_cost = 0
        grad_en_cost = 0
        # evolve 'rho'
        for ii in range(step_max):

            for ll, kraus_gen in enumerate(kraus_gen_list):
                
                # used to add dirivatives to correct jacobians in rho_jac_list for correct kraus_gen
                ind = kraus_gen.param_ind_start
                
                if is_dissip:
                    rho_sup, prob_array, rho_jac_list = apply_dissip_map(kraus_gen, rho_sup, rho_jac_list, ind, opt_props)
                    rho = rho_sup.vector_to_operator(is_copy = True)                    
                    tr = rho.tr()
                    rho_sup = rho_sup*(1/tr)
                else:
                    rho, prob_array, rho_jac_list = apply_krauses(kraus_gen, rho, rho_jac_list, ind, opt_props)
                    tr = rho.tr()
                # Some gates are not normalized since removed blockaded dimensions
                rho = rho*(1/tr)
#                 rho_jac_list = [r*(1/r.tr()) if r != 0 else 0 for r in rho_jac_list]
                prob_array_tot += [prob_array]

                # Fidelity
                if return_F:
                    F_array += [ self.model.get_fidelity(rho) ]
                    E_array += [ self.model.get_energy(rho) ]
                    purity_array += [purity(rho)]

                # Energy
                if ii == mes_step-1 and ll in measure_after_inds:
                    E_cost += self.model.get_energy(rho)
#                     E_cost += iMPS_E(rho, self.model.H_cost)
    
                    F_cost += self.model.get_fidelity(rho)

                    if is_dissip and jac:
                        [ r.vector_to_operator() for r in rho_jac_list ]

                    if jac:
                        grad_en_cost += np.array([self.model.get_energy(jac_rho) for jac_rho in rho_jac_list])
#                         grad_en_cost += np.array([iMPS_E(jac_rho, self.model.H_cost) for jac_rho in rho_jac_list])

            if ii == mes_step-1:
                # stop to calculate jacobian
                rho_cost = rho
                jac = False

        grad_en_cost = (grad_en_cost - E_cost)/eps
        N_m = len(measure_after_inds)
        E_cost /= N_m
        F_cost /= N_m
        
        prob_array_tot = np.array(prob_array_tot).T
        F_array = np.array(F_array)
        E_array = np.array(E_array)
        purity_array = np.real(np.array(purity_array))

        # Approximation of 'prob_array'
        if opt_props.use_probs:
#             y_array_list = prob_array_tot.T[:opt_props.P_N_samp].T
            
#             y_sum_list = [sum(y) for y in y_array_list]
#             max_ind = y_sum_list.index(max(y_sum_list))
#             y_array_min = np.delete(y_array_list, max_ind, 0)
#             y_array = sum(y_array_min)
#     #         y_array = y_array_list[np.argmin(y_array_list.T[-1])]
#             x_array = np.arange(len(y_array))

#             bounds = (0, [np.inf, 1])
#             opt = sp.optimize.curve_fit(f_approx, x_array, y_array, bounds = bounds )
#             a_sum = opt[0][0]
            opt = 0
            a_sum = 0
        else:
            opt = 0
            a_sum = 0
            
        return SimResult(E_cost, F_cost, prob_array_tot, F_array, E_array, purity_array,
                         rho_cost, np.array(grad_en_cost), a_sum = a_sum)
def apply_dissip_map(kraus_gen, rho_sup_in, rho_sup_jac_list, ind, opt_props):
    '''
    return state 'rho_in' affected by a map and the probability for the measuments outcome
    '''
    krauses_sup = kraus_gen.K_list
    map_sup = kraus_gen.map_sup
    
    N_K = len(krauses_sup)
    
    # Proceed rho_jac
    rho_sup_jac_list = [ 
        map_sup* rho_sup_jac
        for rho_sup_jac in rho_sup_jac_list 
    ]
    jac = opt_props.jac
    rho_sup_out = 0
    prob_array = []
    for i in range(N_K):
        
        rho_sup = krauses_sup[i]* rho_sup_in
        if opt_props.use_probs:
            prob_array += [abs(sum(my_vec_to_op(rho_sup).diagonal()))]
            
        rho_sup_out += rho_sup
        
    if jac:
        d_map_sup_list = kraus_gen.d_map_sup_list
        
        rho_sup_jac_add = [ d_map_sup* rho_sup_in for d_map_sup in d_map_sup_list ]
        
        for i in range(len(rho_sup_jac_add)):
            rho_sup_jac_list[i + ind] += rho_sup_jac_add[i]
            
    return rho_sup_out, prob_array, rho_sup_jac_list
def apply_krauses(kraus_gen, rho_in, rho_jac_list, ind, opt_props, ):
    '''
    return state 'rho_in' affected by a Kraus 'K' and the probability for this Kraus
    '''
    K_var_lists = kraus_gen.K_var_lists
    
    K_list = kraus_gen.K_list
    K_dag_list = kraus_gen.K_dag_list
    N_K = len(K_list)
    N_p = kraus_gen.N_free_params
    
    jac = opt_props.jac
    
    if jac:
        # Proceed rho_jac
        for p_ind, rho_jac in enumerate(rho_jac_list):
            # Apply variations of krauses to coresponding variations of state
            if p_ind >= ind and p_ind < ind + N_p:
                if rho_jac ==0: 
                    rho_jac = rho_in
                    
                p_ind_loc = p_ind - ind
                # All variations of krauses for given parameter index 'p_ind_loc'
                K_var_list = [K_var_lists[i][p_ind_loc] for i in range(N_K)]
                rho_jac_list[p_ind] = sum( k*rho_jac*k.dag() for k in K_var_list ) 

            # Apply krauses to others
            else:
                if rho_jac ==0: continue
                rho_jac_list[p_ind] = sum( K_list[i]*rho_jac*K_dag_list[i] for i in range(N_K) )
                
    
    rho_out = 0
    prob_array = []
    for i in range(N_K):
        rho = K_list[i]*rho_in*K_dag_list[i]

        if opt_props.use_probs:
            prob_array += [abs(sum(rho.diagonal()))]
            
        rho_out += rho
    
    return rho_out, prob_array, rho_jac_list

# def apply_krauses(kraus_gen, rho_in, rho_jac_list, ind, opt_props, ):
#     '''
#     return state 'rho_in' affected by a Kraus 'K' and the probability for this Kraus
#     '''
#     dK_lists = kraus_gen.dK_lists
    
#     K_list = kraus_gen.K_list
#     K_dag_list = kraus_gen.K_dag_list
#     N_K = len(K_list)
    
#     # Proceed rho_jac
#     rho_jac_list = [ 
#         sum( K_list[i]*rho_jac*K_dag_list[i] for i in range(N_K) ) if rho_jac !=0 else 0 
#         for rho_jac in rho_jac_list
#     ]
    
#     jac = opt_props.jac
    
#     # Additive for rho_jac_list
#     N_params = kraus_gen.N_free_params
#     rho_jac_add = [0]*N_params
    
#     rho_out = 0
#     prob_array = []
#     for i in range(N_K):
        
#         rho_K = rho_in*K_dag_list[i]
#         rho = K_list[i]*rho_K

#         if opt_props.use_probs:
#             prob_array += [abs(sum(rho.diagonal()))]
            
#         rho_out += rho
        
#         if jac:
#             for j, dK in enumerate(dK_lists[i]):
#                 rho_jac_add[j] += dK*rho_K

#     if jac: 
#         rho_jac_add = [ rho_jac + rho_jac.dag() for rho_jac in rho_jac_add]
#         for i in range(len(rho_jac_add)):
#             rho_jac_list[i + ind] += rho_jac_add[i]

#     return rho_out, prob_array, rho_jac_list
class SimResult(object):
    
    def __init__( self, E, F, P_array, F_array, E_array, purity_array, rho_out, jac, a_sum = 0  ):

        self.E, self.F, self.P_array, self.F_array, self.E_array, self.purity_array, self.rho_out, self.jac, self.a_sum =\
        E, F, P_array, F_array, E_array, purity_array, rho_out, jac, a_sum
class Result(object):
    
    eig_val_2 = None
    F_st_an = None
    E_st_an = None
    with_an = True
    F_100 = None
    E_100 = None
    
    def __init__( self, simulator, res ):
        self.simulator, self.res = simulator, res

        if simulator.system.shape > 62 or not isinstance(simulator, SimulatorKraus):
            self.with_an = False
            
        self.sim_res_N_iter = self.simulator.simulation( self.res.x, return_F = True)
        self.F_N_iter = self.sim_res_N_iter.F
        self.E_N_iter = self.sim_res_N_iter.E
        
#         if isinstance(simulator, SimulatorKraus):
#             self.gen_eigen_props()
        
    def gen_eigen_props(self):
        
        M = reduce(mul,[K_gen.get_sup_map() for K_gen in self.simulator.kraus_gen_list])    
        
        # analytical calculation
        if self.with_an:
            self.eig_st_1, self.eig_val_2 = eigen_props(M)
            self.F_st_an = self.simulator.model.get_fidelity(self.eig_st_1)
            self.E_st_an = self.simulator.model.get_energy( self.eig_st_1)
        
        # steady steate at step 100
        self.sim_res_100 = self.simulator.simulation( self.res.x, return_F = True, step_max = 100)
        self.F_100 = self.sim_res_100.F
        self.E_100 = self.sim_res_100.E
        
        
    def print_results(self):
        print(self.get_message_results())
    
    def get_message_results(self):

        simulator = self.simulator
        opt_props = simulator.opt_props
        system = simulator.system
        model = simulator.model
        
        message =  "State:          " + model.TAG + '\n' +\
        "N sys., N aux.: " + str(system.N_sys) + ', ' + str(system.N_aux) + '\n' +\
        "N layers:       " + str(simulator.N_layers) + '\n' +\
        "N params:       " + str(simulator.N_free_params) + '\n' +\
        "N iter.:        " + str(simulator.N_iter) + '\n' +\
        "Tol. rel.:      " + str(opt_props.tol_rel) + '\%\n' +\
        "Method:         " + str(opt_props.method) + '\n' +\
        "nfev, nit:      " + str(self.res.nfev) + ', '+ str(self.res.nit) + '\n' +\
        "E max.:         " + "%.2f" % model.E_max +'\n' +\
        "E min:          " + "%.2f" % model.E_min
        
        if self.eig_val_2 is not None:
            message += '\n' +\
            "Sec. eig. val.: " + "%.2f" % self.eig_val_2 +'\n' +\
            "F st. an.:      " + "%.3f" % self.F_st_an +'\n' +\
            "E st. an.:      " + "%.2f" % self.E_st_an
                       
            
        if self.F_100 is not None:
            message += '\n' +\
            "F(100 cycles):  " + "%.3f" % self.F_100 +'\n' +\
            "E(100 cycles):  " + "%.2f" % self.E_100
                       

        message += '\n' +\
        "F(N_iter cycles):  " + "%.3f" % self.F_N_iter +'\n' +\
        "E(N_iter cycles):  " + "%.2f" % self.E_N_iter
        
        return message
def array_to_fast_csr(M):
    M_scr = sp.sparse.csr_matrix(M)
    return fastsparse.fast_csr_matrix([M_scr.data, M_scr.indices, M_scr.indptr])
def red_and_blue_gates(dim_phonon, inds = [0,1], dim_sys = 2, offset = 0):
    '''
    list of red and blue palse coupling a phonon mode (1-st mode) with an ions (2-nd mode)
    '''
    
    a = destroy(dim_phonon, offset)
    
    g_blue = tensor(a,fock(dim_sys,inds[0])*fock(dim_sys,inds[1]).dag())
    g_red = tensor(a,fock(dim_sys,inds[1])*fock(dim_sys,inds[0]).dag())
    
    g_blue = g_blue + g_blue.dag()
    g_red = g_red + g_red.dag()
    
    return g_blue, g_red

def red_and_blue_gates_full(dim_aux, sys_props, offset = 0):
    '''
    list of red and blue palse coupling ions with a phonon mode
    '''
    I = sys_props.I_sys_list[0]
    N_sys = sys_props.N_sys
    N_modes = sys_props.N_modes
    
    a = destroy(dim_aux, offset)
#     g_blue = 0
#     g_red = 0
#     for i in range(dim_aux-1):
#         g_blue += (i+1)**0.5 * tensor(fock(dim_aux,i),fock(2,0)) * tensor(fock(dim_aux,i+1),fock(2,1)).dag()
#         g_red += (i+1)**0.5 * tensor(fock(dim_aux,i),fock(2,1)) * tensor(fock(dim_aux,i+1),fock(2,0)).dag()
    
    g_blue = tensor(a,fock(2,0)*fock(2,1).dag())
    g_red = tensor(a,fock(2,1)*fock(2,0).dag())
    
    g_blue = tensor( [g_blue + g_blue.dag()] + [I]*(N_sys-1) )
    g_blue_list = [g_blue]
    
    g_red = tensor( [g_red + g_red.dag()] + [I]*(N_sys-1) )
    g_red_list = [g_red]
    for i in range(2, N_modes):
        order = np.arange(N_modes)
        order[1], order[i] = order[i], order[1]
        g_blue_list += [g_blue.permute(order)]
        g_red_list += [g_red.permute(order)]
    
    return g_blue_list, g_red_list

def red_and_blue_gates_full_2(dim_aux, N_sys, offset = 0):
    '''
    list of red and blue palse coupling ions with a phonon mode
    '''
    I = identity(2)
    N_modes = N_sys+1
    a = destroy(dim_aux, offset)
#     g_blue = 0
#     g_red = 0
#     for i in range(dim_aux-1):
#         g_blue += (i+1)**0.5 * tensor(fock(dim_aux,i),fock(2,0)) * tensor(fock(dim_aux,i+1),fock(2,1)).dag()
#         g_red += (i+1)**0.5 * tensor(fock(dim_aux,i),fock(2,1)) * tensor(fock(dim_aux,i+1),fock(2,0)).dag()
    
    g_blue = tensor(a,fock(2,0)*fock(2,1).dag())
    g_red = tensor(a,fock(2,1)*fock(2,0).dag())
    
    g_blue = tensor( [g_blue + g_blue.dag()] + [I]*(N_sys-1) )
    g_blue_list = [g_blue]
    
    g_red = tensor( [g_red + g_red.dag()] + [I]*(N_sys-1) )
    g_red_list = [g_red]
    for i in range(2, N_modes):
        order = np.arange(N_modes)
        order[1], order[i] = order[i], order[1]
        g_blue_list += [g_blue.permute(order)]
        g_red_list += [g_red.permute(order)]
    
    return g_blue_list, g_red_list
def get_wide_pulses_H(intens_coef, phi, dim_phonon, offset = 0):
    '''
    Wide waist pulses for an ions array
    H = a*{ S_j + intens_coef*[S_{j-1}*e^(-1j*phi) + S_{j+1}*e^(1j*phi)] } + H.C.
    '''
    a = destroy(dim_phonon, offset)
    c_l = intens_coef * cmath.exp(-1j*phi)
    c_r = intens_coef * cmath.exp(1j*phi)
    
    I = identity(2)
    
    g_blue_l = tensor([S_m, I]) + c_r * tensor([I, S_m])
    g_blue_r = c_l * tensor([S_m, I]) + tensor([I, S_m])
    g_blue_m = c_l * tensor([S_m, I, I]) + tensor([I, S_m, I]) + c_r * tensor([I, I, S_m])
    
    g_blue_l = tensor(a, g_blue_l)
    g_blue_r = tensor(a, g_blue_r)
    g_blue_m = tensor(a, g_blue_m)
    
    g_blue_l = g_blue_l + g_blue_l.dag()
    g_blue_r = g_blue_r + g_blue_r.dag()
    g_blue_m = g_blue_m + g_blue_m.dag()
    
#     g_red = tensor(a,S_p)
    
#     g_blue = g_blue + g_blue.dag()
#     g_red = g_red + g_red.dag()
    
    return g_blue_l, g_blue_r, g_blue_m
def get_H_MS( S, N_mode, inds = []):
    '''
    Create 'N_mode' Molmer sorensen Hamiltonian Sum_{nn,ll>nn}(S_nn * S_ll) with S a Pauli matrix
    '''
    if len(inds) == 0:
        inds = range(N_mode)
    
    S_list = [];
    for ii in inds:
        S_list += [make_multimode_oper(S, N_mode, [ii])]

    N = len(inds)
    
    H_mc = 0;
    for nn in range(N):
        for ll in range(nn+1,N):
            H_mc += S_list[nn] * S_list[ll]

    return H_mc
def get_H_J( N_mode, B):
    '''
    Create 'N_mode' Molmer sorensen Hamiltonian Sum_{nn,ll>nn}(S_nn * S_ll) with S a Pauli matrix
    '''
    
    J = get_J(N_mode)
    
    Z_list = [];
    X_list = [];
    for ii in range(N_mode):
        Z_list += [make_multimode_oper(sigmaz(), N_mode, [ii])]
        X_list += [make_multimode_oper(sigmax(), N_mode, [ii])]
    
    H_J = 0;
    for nn in range(N_mode):
        for ll in range(nn+1,N_mode):
            H_J += X_list[nn] * X_list[ll] * J[nn,ll]

    H_Z = sum(Z_list)
    
    return H_J, H_Z, H_J + B * H_Z
def get_my_Hij(N_mode, B=1, alpha = 1.34, J0=1):
    Sz_list = [make_multimode_oper(-S_z, N_mode, [ii]) for ii in range(N_mode)]
    Sp_list = [make_multimode_oper(S_p, N_mode, [ii]) for ii in range(N_mode)]
    Sm_list = [make_multimode_oper(S_m, N_mode, [ii]) for ii in range(N_mode)]
    
    Hij = 0
    for i in range(N_mode-1):
        for j in range(i+1,N_mode):
            Hij += J0/abs(i-j)**alpha*(Sp_list[i]*Sm_list[j] + Sm_list[i]*Sp_list[j])
    Hij += B* sum(Sz_list)
    
    return Hij
def get_J(N):
    return sp.io.loadmat('libs/J_matrix/J'+ str(N) +'.mat')['Jij']
def get_schw_H_revers(N_mode, J = 1, w = 1, m = -0.5, e0 = 0):
    '''
    Return 'N_mode' Schwinger Hamiltonian
    '''
    
    Sz_list = [];
    Sp_list = [];
    Sm_list = [];
    for ii in range(N_mode)[::-1]:
        Sz_list += [make_multimode_oper(-S_z, N_mode, [ii])]
        Sp_list += [make_multimode_oper(S_p, N_mode, [ii])]
        Sm_list += [make_multimode_oper(S_m, N_mode, [ii])]
    
    Hs = w * sum( [ (Sp_list[n] * Sm_list[n+1] + Sp_list[n+1] * Sm_list[n]) for n in range(N_mode-1) ] )
   
    Ln_array = [ e0 -1./2 * sum( Sz_list[l] -(-1)**l for l in range(n+1)) 
                for n in range(N_mode-1) ]
    
    HJ = J * sum( L * L for L in Ln_array)
    
    Hz_m = -m/2 * sum( [ (-1)**n * Sz_list[n] for n in range(N_mode) ])
    
    return Hs + HJ + Hz_m

def get_schw_H(N_mode, J = 1, w = 1, m = -0.5, e0 = 0):
    '''
    Return 'N_mode' Schwinger Hamiltonian
    '''
    
    Sz_list = [];
    Sp_list = [];
    Sm_list = [];
    for ii in range(N_mode):
        Sz_list += [make_multimode_oper(S_z, N_mode, [ii])]
        Sp_list += [make_multimode_oper(S_p, N_mode, [ii])]
        Sm_list += [make_multimode_oper(S_m, N_mode, [ii])]
    
    Hs = w * sum( [ (Sp_list[n] * Sm_list[n+1] + Sp_list[n+1] * Sm_list[n]) for n in range(N_mode-1) ] )

   
    Ln_array = [ e0 - 1./2 * sum( [ ( Sz_list[l] -(-1)**l ) for l in range(n) ]) for n in range(1, N_mode) ]

    HJ = J * sum( [ L * L for L in Ln_array] )
    
    Hz_m = -m/2 * sum( [ (-1)**n * Sz_list[n] for n in range(N_mode) ])

    return Hs + HJ + Hz_m

def get_schw_H_half_invert(N_mode):
    H = get_schw_H(N_mode)
    if N_mode % 2 == 1:
        raise Exception("N_mode has to be even")
        
    N2 = N_mode//2
    I = identity(2)
    inv = tensor([I]*N2+[S_x]*N2)
    return inv * H * inv
def get_GHZ_H(N_mode):
    '''
    Cost H for |0000...>+|1111...>
    '''
    Sz_list = [make_multimode_oper(S_z, N_mode, [ii]) for ii in range(N_mode)]
    
    return sum( [ -(Sz_list[n] * Sz_list[n+1]) for n in range(N_mode-1) ] ) - tensor([S_x]*N_mode)

def get_GHZ_antiferromag_H(N_mode):
    '''
    Cost H for |0101...>+|1010...>
    '''
    Sz_list = [make_multimode_oper(S_z, N_mode, [ii]) for ii in range(N_mode)]
    
    return sum( [ (Sz_list[n] * Sz_list[n+1]) for n in range(N_mode-1) ] ) - tensor([S_x]*N_mode)
def get_ising_H(N_mode):
    '''
    Return 'N_mode' Ising Hamiltonian
    '''
    
    Sz_list = [];
    Sx_list = [];
    for ii in range(N_mode)[::-1]:
        Sz_list += [make_multimode_oper(sigmaz(), N_mode, [ii])]
        Sx_list += [make_multimode_oper(sigmax(), N_mode, [ii])]
    
    H_xx = sum( Sx_list[i] * Sx_list[i+1] for i in range(N_mode-1) )
    H_z = sum( Sz_list )
    
    
    return H_xx + H_z
def get_cl_ising_H(N_mode, l, h):
    '''
    Return 'N_mode' Cluster Ising Hamiltonian
    '''
    
    Sz_list = []
    Sx_list = []
    Sy_list = []
    for ii in range(N_mode)[::-1]:
        Sz_list += [make_multimode_oper(sigmaz(), N_mode, [ii])]
        Sx_list += [make_multimode_oper(sigmax(), N_mode, [ii])]
        Sy_list += [make_multimode_oper(sigmay(), N_mode, [ii])]
    
    H_xzx = sum( Sx_list[i] * Sz_list[i+1] * Sx_list[i+2] for i in range(N_mode-2) )
    H_yy = sum( Sy_list[i] * Sy_list[i+1] for i in range(N_mode-1) )
    H_z = sum( Sz_list )
    
    
    return -H_xzx - l * H_yy + h * H_z
def eigen_props(E_sup):
    '''
    return the firs eigen state and the difference between the first and the second eigenvalue
    '''
    
    if isinstance(E_sup, MyQobj):
        E_sup = E_sup.data
    
    if not isinstance(E_sup, Qobj):
        E_sup = Qobj(E_sup)
        
    vals, vecs = E_sup.eigenstates()
    vals = np.abs(vals)

    e_val_2 = vals[-2]

    e_st_1_vec = vecs[-1]
    e_st_1_vec.dims = [[[int(e_st_1_vec.shape[0]**0.5)]]*2,[1]]

    e_st_1 = vector_to_operator(e_st_1_vec)


    for i in range(e_st_1.shape[0]):
        element = e_st_1.data[i,i]

        if abs(np.real_if_close(element))!=0:
            break
    if element!=0:
        e_st_1 = (e_st_1 / element).unit()
        
    return e_st_1, e_val_2
def get_SSH_H(N_mode, t_m, t_p = 1, B=0):
    '''
    Return 'N_mode' SSH Hamiltonian
    '''
    Sx_list = [make_multimode_oper(S_x, N_mode, [ii]) for ii in range(N_mode)]
    Sy_list = [make_multimode_oper(S_y, N_mode, [ii]) for ii in range(N_mode)]
    Sz_list = [make_multimode_oper(S_z, N_mode, [ii]) for ii in range(N_mode)]
    
    H = sum( [ (Sx_list[n] * Sx_list[n+1] + Sy_list[n] * Sy_list[n+1])*(t_p+(-1)**n*t_m) for n in range(N_mode-1) ] ) +\
    B*(Sz_list[0]-Sz_list[-1])
    
    return H
def get_W_H(N_mode):
    '''
    Return 'N_mode' parent Hamiltonian wor W state
    '''
    Sx_list = [make_multimode_oper(S_x, N_mode, [ii]) for ii in range(N_mode)]
    Sy_list = [make_multimode_oper(S_y, N_mode, [ii]) for ii in range(N_mode)]
    Sz_list = [make_multimode_oper(S_z, N_mode, [ii]) for ii in range(N_mode)]
    
    H = - sum( [ (Sx_list[n] * Sx_list[n+1] + Sy_list[n] * Sy_list[n+1]) for n in range(N_mode-1) ] )
    H -= Sx_list[N_mode-1] * Sx_list[0] + Sy_list[N_mode-1] * Sy_list[0]
    H += (sum(Sz_list[n] for n in range(N_mode) ) + N_mode - 2)**2
    
#     H = sum( [ (Sx_list[n] * Sx_list[n+1]+Sy_list[n] * Sy_list[n+1]) for n in range(N_mode-1) ] )
#     H += 1*sum(Sx_list[n] for n in range(N_mode) )
    
    return H
def get_MG_H(N_mode):
    '''
    Return 'N_mode' parent Hamiltonian wor W state
    '''
    Sx_list = [make_multimode_oper(S_x, N_mode, [ii]) for ii in range(N_mode)]
    Sy_list = [make_multimode_oper(S_y, N_mode, [ii]) for ii in range(N_mode)]
    Sz_list = [make_multimode_oper(S_z, N_mode, [ii]) for ii in range(N_mode)]
    
    H = 0
    for S_list in [Sx_list, Sy_list, Sz_list]:
        H += 2*sum( S_list[n] * S_list[n+1] for n in range(N_mode-1) )
        H += sum( S_list[n] * S_list[n+2] for n in range(N_mode-2) )
    
    for S_list in [Sx_list, Sy_list, Sz_list]:
        H += 2*S_list[N_mode-1] * S_list[0]
        H += S_list[N_mode-2] * S_list[0]
        H += S_list[N_mode-1] * S_list[1]
        
        
    return H
def get_clust_H(N_mode, b = 0):
    '''
    Return 'N_mode' parent Hamiltonian wor a cluster state
    '''
    Sx_list = [make_multimode_oper(S_x, N_mode, [ii]) for ii in range(N_mode)]
    Sy_list = [make_multimode_oper(S_y, N_mode, [ii]) for ii in range(N_mode)]
    Sz_list = [make_multimode_oper(S_z, N_mode, [ii]) for ii in range(N_mode)]
    
    H = 0
    for i in range(N_mode-2):
#         H -= Sx_list[i] * Sz_list[i+1] * Sx_list[i+2]
        H -= Sx_list[i] * Sz_list[i+1] * Sx_list[i+2]
    H += b*(Sz_list[0] + Sz_list[-1])
    return H
def get_squeez_H(N_mode):
    '''
    Return 'N_mode' parent Hamiltonian wor a squeesed state
    '''
    Sx_list = [make_multimode_oper(S_x, N_mode, [ii]) for ii in range(N_mode)]
    Sy_list = [make_multimode_oper(S_y, N_mode, [ii]) for ii in range(N_mode)]
    Sz_list = [make_multimode_oper(S_z, N_mode, [ii]) for ii in range(N_mode)]
    
    J_z = sum(Sz_list)
    J_x = sum(Sx_list)
    J_y = sum(Sy_list)
    
    H = -2*J_x**2 - 2*J_y**2 - J_z**2
    
    return H
def get_AKLT_H_3(N_mode):
    '''
    Return 'N_mode' AKLT Hamiltonian
    '''
    S3_x_list = [make_multimode_oper(S3_x, N_mode, [ii], dim = 3) for ii in range(N_mode)]
    S3_y_list = [make_multimode_oper(S3_y, N_mode, [ii], dim = 3) for ii in range(N_mode)]
    S3_z_list = [make_multimode_oper(S3_z, N_mode, [ii], dim = 3) for ii in range(N_mode)]
    
    H = 0
    for n in range(N_mode-1):
        SS = sum(S_list[n]*S_list[n+1] for S_list in [S3_x_list,S3_y_list,S3_z_list] )
        H += SS
        H += 1/3*SS**2
    
    return H

def get_AKLT_H_2(N_mode, l = 0):
    '''
    Return 'N_mode' AKLT Hamiltonian
    '''
    N_mode_3 = N_mode//2
    H = get_AKLT_H_3(N_mode_3)
    map_3_to_2_mult_mod = tensor([map_3_to_2]*N_mode_3)
    
    return map_3_to_2_mult_mod*H*map_3_to_2_mult_mod.dag()


def get_AKLT_H_2_2(N_mode, l = 0, b = 0):
    '''
    Return 'N_mode' AKLT Hamiltonian
    '''
    N_mode_3 = N_mode//2
    I = identity(2)
    S3_x_2= (tensor(I, S_x)+tensor(S_x, I))/2
    S3_y_2= (tensor(I, S_y)+tensor(S_y, I))/2
    S3_z_2= (tensor(I, S_z)+tensor(S_z, I))/2

    S3_x_list = [make_multimode_oper(S3_x_2, N_mode-1, [2*ii], dim = 2) for ii in range(N_mode_3)]
    S3_y_list = [make_multimode_oper(S3_y_2, N_mode-1, [2*ii], dim = 2) for ii in range(N_mode_3)]
    S3_z_list = [make_multimode_oper(S3_z_2, N_mode-1, [2*ii], dim = 2) for ii in range(N_mode_3)]
    
    H = 0
    for n in range(N_mode_3-1):
        SS = sum(S_list[n]*S_list[n+1] for S_list in [S3_x_list,S3_y_list,S3_z_list] )
        H += SS
        H += 1/3*SS**2
    
    d = (1+tensor(S_x, S_x) + tensor(S_y, S_y) + tensor(S_z, S_z))/2
#     H -= l * (tensor(d, I,I) + tensor(I,I,d))
    
    
    
#     H -= l * sum( 
#         make_multimode_oper(d, N_mode-1, [2*ii], dim = 2) 
#         for ii in range(N_mode_3)
#     )

    Sz_list = [make_multimode_oper(S_z, N_mode, [ii]) for ii in range(N_mode)]
    H +=b*(Sz_list[0] - Sz_list[-1] + Sz_list[1] - Sz_list[-2])
    
    H -= l * sum( 
        make_multimode_oper(tensor(S, S), N_mode-1, [2*ii], dim = 2) 
        for S in [S_x, S_y, S_z] for ii in range(N_mode_3)
    )
    H += l * N_mode_3
    return H
def get_H_Hald_qub(N_mode, J = 1, Delt = 1, D2 = 1, l = 0):
    
    N_mode_3 = N_mode//2
    I = identity(2)
    S3_x_2= (tensor(I, S_x)+tensor(S_x, I))/2
    S3_y_2= (tensor(I, S_y)+tensor(S_y, I))/2
    S3_z_2= (tensor(I, S_z)+tensor(S_z, I))/2

    S3_x_list = [make_multimode_oper(S3_x_2, N_mode-1, [2*ii], dim = 2) for ii in range(N_mode_3)]
    S3_y_list = [make_multimode_oper(S3_y_2, N_mode-1, [2*ii], dim = 2) for ii in range(N_mode_3)]
    S3_z_list = [make_multimode_oper(S3_z_2, N_mode-1, [2*ii], dim = 2) for ii in range(N_mode_3)]
    
    H = 0
    for n in range(N_mode_3-1):
        H += sum(S_list[n]*S_list[n+1] for S_list in [S3_x_list,S3_y_list] )
        H += Delt*S3_z_list[n]*S3_z_list[n+1]
    
    H += D2*sum(S3_z*S3_z for S3_z in S3_z_list)
    H *= J
    
    H -= l * sum( 
        make_multimode_oper(tensor(S, S), N_mode-1, [2*ii], dim = 2) 
        for S in [S_x, S_y, S_z] for ii in range(N_mode_3)
    )
    H += l * N_mode_3
    return H
def get_H_Hald_old(N, J = 1, Delt = 1, D2 = 1):
# def get_H_Hald(N, J = 10, Delt = 100, D2 = 0.1):
#     J = 1
#     Delt = 10
#     D2 = 1
    Sr_x_array = [make_multimode_oper(Sr_x, N, [i], dim = 3) for i in range(N)]
    Sr_y_array = [make_multimode_oper(Sr_y, N, [i], dim = 3) for i in range(N)]
    Sr_z_array = [make_multimode_oper(Sr_z, N, [i], dim = 3) for i in range(N)]
    
    SxSx = sum( Sr_x_array[i] * Sr_x_array[i+1] for i in range(N-1))
    SySy = sum( Sr_y_array[i] * Sr_y_array[i+1] for i in range(N-1))
    SzSz = sum( Sr_z_array[i] * Sr_z_array[i+1] for i in range(N-1))
    
    Sz2 = sum(s*s for s in Sr_z_array)
    
    return J * (SxSx + SySy + Delt * SzSz) + D2 * Sz2
def get_H_Hald(N, J = 1, Delt = 3, D2 = 0.5):
# def get_H_Hald(N, J = 10, Delt = 100, D2 = 0.1):
#     J = 1
#     Delt = 3
#     D2 = 0.5
    dim = Sr_x.dims[0][0]
    Sr_x_array = [make_multimode_oper(Sr_x, N, [i], dim = dim) for i in range(N)]
    Sr_y_array = [make_multimode_oper(Sr_y, N, [i], dim = dim) for i in range(N)]
    Sr_z_array = [make_multimode_oper(Sr_z, N, [i], dim = dim) for i in range(N)]
    
    SxSx = sum( Sr_x_array[i] * Sr_x_array[i+1] for i in range(N-1))
    SySy = sum( Sr_y_array[i] * Sr_y_array[i+1] for i in range(N-1))
    SzSz = sum( Sr_z_array[i] * Sr_z_array[i+1] for i in range(N-1))
    
    Sz2 = sum(s*s for s in Sr_z_array)
    
    Se_sum = sum(make_multimode_oper(Se, N, [i], dim = dim) for i in range(N))
    return J * (SxSx + SySy + Delt * SzSz) + D2 * Sz2 + 0*Se_sum
def get_H_Hald_qubits(N_sys, trans_map, J = 1, Delt = 1, D2 = 1):
    '''
    Haldane Hamiltonian on spin 1/2 system
    '''
    H_Hald = get_H_Hald(N_sys, J, Delt, D2)
    
    trans_N = tensor([trans_map]*N_sys)
    return trans_N.dag() * H_Hald * trans_N
def get_dims_to_del(coords, max_bound):
    '''
    Dimensions which are not populated
    '''
    N = len(coords)
    S_z_array = [ make_multimode_oper(S_z, N, [i]) for i in range(N) ]
    
    # Blockade Hamiltonian 
    H_ij = 0
    for i in range(N):
        x_i, y_i = coords[i]
        for j in range(i+1,N):
            x_j, y_j = coords[j]

            d = (x_j - x_i)**2 + (y_j - y_i)**2

            H_ij += (1+S_z_array[i])*(1+S_z_array[j])/d**3

    return H_ij.data.indices[H_ij.data.data>max_bound]
def map_ryd(H, inds_rows, inds_cols = None):
    
    if inds_cols is None:
        inds_cols = inds_rows
        
    if len(inds_rows) != 0 and H.shape[0]!=1:
        if np.max(inds_rows) > H.shape[0]-1:
            raise ValueError
        if isinstance(H, np.ndarray):
            H = np.delete(H, inds_rows, 0)
        else:
            inds_rows_to_save = [i for i in range(H.shape[0]) if i not in inds_rows]
            H = H[inds_rows_to_save,:]
    
    if len(inds_cols) != 0 and H.shape[1]!=1:
        if np.max(inds_cols) > H.shape[1]-1:
            raise ValueError
        if isinstance(H, np.ndarray):
            H = np.delete(H, inds_cols, 1)
        else:
            inds_cols_to_save = [i for i in range(H.shape[1]) if i not in inds_cols]
            H = H[:, inds_cols_to_save]
    
    return H
def get_H_Ryd_list(coords_sys, coords_aux, max_bound = np.inf):
    '''
    Resourse Hamiltonians for Rydberg atoms: 
    'coords' - coordinates of rydberg atoms.
    Dimensionse with elements above 'max_bound' are deleted sinse they are blockaded, 
    so the result Hamiltonians have redused dimension.
    '''
    
    coords = coords_aux + coords_sys
    
    # Number of modes
    N_sys_modes = len(coords_sys)
    N_aux_modes = len(coords_aux)
    N = N_sys_modes + N_aux_modes
    
    # Single qubit rotations
    S_x_array = [ make_multimode_oper(S_x, N, [i]) for i in range(N) ]
    S_z_array = [ make_multimode_oper(S_z, N, [i]) for i in range(N) ]

    # Blockade Hamiltonian 
    H_ij = 0
    for i in range(N):
        x_i, y_i = coords[i]
        for j in range(i+1,N):
            x_j, y_j = coords[j]

            d = (x_j - x_i)**2 + (y_j - y_i)**2

            H_ij += (1+S_z_array[i])*(1+S_z_array[j])/d**3
    
    # Delete dimensions which are not populated
    dims_to_del = H_ij.data.indices[H_ij.data.data>max_bound]
    
    H_ij = map_ryd(H_ij.data, dims_to_del)
    S_x_array = [ 0.5 * map_ryd(S_x.data, dims_to_del) for S_x in S_x_array ]
    S_z_array = [ -map_ryd(S_z.data, dims_to_del) for S_z in S_z_array ]
    
    S_x_aux_list = S_x_array[:N_aux_modes]
    S_z_aux_list = S_z_array[:N_aux_modes]
    
    S_x_sys_list = [ sum(S_x_array[j] for j in [N_aux_modes+2*i,N_aux_modes+2*i+1]) for i in range(N_sys_modes//2 )]
    S_z_sys_list = [ sum(S_z_array[j] for j in [N_aux_modes+2*i,N_aux_modes+2*i+1]) for i in range(N_sys_modes//2 )]
    
    return H_ij, S_x_sys_list, S_z_sys_list, S_x_aux_list, S_z_aux_list
def get_H_ryd_block(coords, S_z = S_z):
    N = len(coords)    
    S_z_array = [ make_multimode_oper(S_z, N, [i], dim=get_q_dim_list(S_z)) for i in range(N) ]
    
    H_ij = 0
    for i in range(N):
        x_i, y_i = coords[i]
        for j in range(i+1,N):
            x_j, y_j = coords[j]

            d = (x_j - x_i)**2 + (y_j - y_i)**2

            H_ij += (1+S_z_array[i])*(1+S_z_array[j])/d**3
    
    return H_ij
def get_H_Ryd_list_aux_lvls(coords_sys, max_bound = np.inf):
    '''
    Resourse Hamiltonians for Rydberg atoms: 
    'coords' - coordinates of rydberg atoms.
    Dimensionse with elements above 'max_bound' are deleted sinse they are blockaded, 
    so the result Hamiltonians have redused dimension.
    '''
    
    coords = coords_sys
    
    # Number of modes
    N_sys_modes = len(coords_sys)
    N = N_sys_modes
    
    N_dim = 3
    # Single qubit rotations
#     X3 = Sr_x
    S_x_aux_list = [ make_multimode_oper(X3_aux, N, [i], dim = N_dim) for i in range(N) ]
    S_z_aux_list = [ make_multimode_oper(Z3_aux, N, [i], dim = N_dim) for i in range(N) ]
    
    S_x_sys_list = [ make_multimode_oper(X3_sys, N, [i], dim = N_dim) for i in range(N) ]
    S_z_sys_list = [ make_multimode_oper(Z3_sys, N, [i], dim = N_dim) for i in range(N) ]
    S_z_block_list = [ make_multimode_oper(Z3_block, N, [i], dim = N_dim) for i in range(N) ]

    # Blockade Hamiltonian 
    H_ij = 0
    for i in range(N):
        x_i, y_i = coords[i]
        for j in range(i+1,N):
            x_j, y_j = coords[j]

            d = (x_j - x_i)**2 + (y_j - y_i)**2

            H_ij += (1+S_z_block_list[i])*(1+S_z_block_list[j])/d**3
    
    # Delete dimensions which are not populated
    dims_to_del = H_ij.data.indices[H_ij.data.data>max_bound]
    
    H_ij = map_ryd(H_ij.data, dims_to_del)
    
    S_x_aux_list = [ 0.5 * map_ryd(S_x.data, dims_to_del) for S_x in S_x_aux_list ]
    S_x_sys_list = [ 0.5 * map_ryd(S_x.data, dims_to_del) for S_x in S_x_sys_list ]
    
    S_z_aux_list = [ -map_ryd(S_z.data, dims_to_del) for S_z in S_z_aux_list ]
    S_z_sys_list = [ -map_ryd(S_z.data, dims_to_del) for S_z in S_z_sys_list ]
    
    
    S_x_sys_list = [ sum(S_x_sys_list[j] for j in [2*i,2*i+1]) for i in range(N_sys_modes//2 )]
    S_z_sys_list = [ sum(S_z_sys_list[j] for j in [2*i,2*i+1]) for i in range(N_sys_modes//2 )]
    
    S_x_aux_list = [ sum(S_x_aux_list[j] for j in [2*i,2*i+1]) for i in range(N_sys_modes//2 )]
    S_z_aux_list = [ sum(S_z_aux_list[j] for j in [2*i,2*i+1]) for i in range(N_sys_modes//2 )]
    
    return H_ij, S_x_sys_list, S_z_sys_list, S_x_aux_list, S_z_aux_list
X3_sys = Qobj([[0,1,0],[1,0,0],[0,0,0]])
X3_aux = Qobj([[0,0,1],[0,0,0],[1,0,0]])
X3_sys
X3_aux
N_dim = 3
Z3_block = -fock_dm(N_dim, 0) + fock_dm(N_dim, 1) + fock_dm(N_dim, 2)
# Z3_sys = -fock_dm(N_dim, 0) + fock_dm(N_dim, 1)
Z3_block
N_dim = 3
Z3_sys = -fock_dm(N_dim, 0) + fock_dm(N_dim, 1) - fock_dm(N_dim, 2)
# Z3_sys = -fock_dm(N_dim, 0) + fock_dm(N_dim, 1)
Z3_sys
Z3_aux = -fock_dm(N_dim, 0) - fock_dm(N_dim, 1) + fock_dm(N_dim, 2)
# Z3_aux = -fock_dm(N_dim, 0) + fock_dm(N_dim, 2)
Z3_aux
def get_dims_to_del_aux_lvls(coords, max_bound):
    '''
    Dimensions which are not populated
    '''
    N = len(coords)
    N_dim = 3
    
#     S_z_array = [ make_multimode_oper(Z3_sys, N, [i], dim=N_dim) for i in range(N) ]
    S_z_block_list = [ make_multimode_oper(Z3_block, N, [i], dim = N_dim) for i in range(N) ]
    
    # Blockade Hamiltonian 
    H_ij = 0
    for i in range(N):
        x_i, y_i = coords[i]
        for j in range(i+1,N):
            x_j, y_j = coords[j]

            d = (x_j - x_i)**2 + (y_j - y_i)**2

            H_ij += (1+S_z_block_list[i])*(1+S_z_block_list[j])/d**3

    return H_ij.data.indices[H_ij.data.data>max_bound]
def params_map_sym_CI(N_lays):
    
    N_params = N_lays//2+1
    params_inds = list(range(N_params))
    params_map = params_inds + params_inds[:-1][::-1]
    params_map_0 = np.array(params_map)

    params_map = [ np.where(params_map_0 == i)[0] for i in range(N_params)]
    
    return params_map


def params_map_sym_Hald(N_layers, N_gate_params):

    params_map = []

    K = N_layers//2+1

    lay_inds_0 = list(range(K))
    lay_inds = lay_inds_0 + lay_inds_0[:-1][::-1]

    N_params = K * N_gate_params
    params_inds = list(range(N_params))
    for i in lay_inds:
        params_map += params_inds[i*N_gate_params:(i+1)*N_gate_params]
    params_map_0 = np.array(params_map)

    params_map = [ np.where(params_map_0 == i)[0] for i in range(N_params)]
    
    return params_map


def position_Hald( N_sys, b = 0.5, d = 0.1, plot = False, plot_size = 3):
    '''
    Coordinates for system and anncilla rydbergs and their gates combineations.
    b - verdical distance between system qubits, d - vertical distance between sys. and aux. atoms.
    '''
    
    coords_sys = [[i//2,divmod(i,2)[1]*b] for i in range(N_sys*2)]
    combine_sys = [[i,N_sys-i-1] for i in range(N_sys//2)]
    if divmod(N_sys,2)[1] == 1:
        combine_sys += [[N_sys//2]]

    coords_aux = [[i//2,-d + divmod(i,2)[1]*(b+2*d)] for i in range(N_sys*2)]
    combine_aux = [[i*2,i*2+1,N_sys*2-i*2-2,N_sys*2-i*2-1] for i in range(N_sys//2)]
    if divmod(N_sys,2)[1] == 1:
        combine_aux += [[N_sys-1, N_sys]]
    
    if plot:
         plot_scheme(coords_sys, coords_aux, plot_size)
        
    return coords_sys, coords_aux, combine_sys, combine_aux
def plot_scheme(coords_sys, coords_aux, plot_size):
    fig, ax = plt.subplots()
    fig.set_size_inches(plot_size,plot_size/2.)
    _plot_scheme(ax, coords_sys, coords_aux, with_frame=False, with_ax = True)
def get_H_ions_phonons(system, mu):
    
    N_sys = system.N_sys_modes
    N_aux = system.N_aux
    dim_sys = system.dim_sys
    dim_aux = system.dim_aux

    if dim_sys != 2:
        raise ValueError("dim_sys != 2")

    if N_sys != N_aux:
        raise ValueError("N_sys != N_aux")
        
    if N_sys != 2:
        raise NotImplementedError
    
    I_a = identity(dim_aux)
    a = destroy(dim_aux)
    A = a + a.dag()
    
    mu_c = mu / 2**0.5
    mu_r = mu * (4/3)**(1/4)
    
    a_exp = (1j * mu_c * A).expm()
    b_exp = (1j/2 * mu_r * A).expm()
    
    H_1 = tensor([a_exp, b_exp, S_p])
    H_1 += H_1.dag()
    
    H_2 = tensor([a_exp, b_exp.dag(), S_p])
    H_2 += H_2.dag()
    
    H_free = tensor( a.dag() * a, I_a) + tensor(I_a, 3**0.5 * a.dag() * a)
    
    return H_1, H_2, H_free
def get_H_ions_phonons_full(system, mu0, delta_1 = 0, delta_2 = 0):
    
    N_sys = system.N_sys_modes
    N_aux = system.N_aux
    dim_sys = system.dim_sys
    dim_aux = system.dim_aux

    if dim_sys != 2:
        raise ValueError("dim_sys != 2")

    if N_sys != N_aux:
        raise ValueError("N_sys != N_aux")
        
    if N_sys != 2:
        raise NotImplementedError
    
    I_a = identity(dim_aux)
    a = destroy(dim_aux)
    A = a + a.dag()
    
    I_s = identity(dim_sys)
    
    H_list = []
    for mu in [mu0, -mu0]:
        mu_c = mu / 2**0.5
        mu_r = mu * (4/3)**(1/4)

        a_exp = (1j * mu_c * A).expm()
        b_exp = (1j/2 * mu_r * A).expm()

        H_1 = tensor([a_exp, b_exp, S_p, I_s])
        H_1 += H_1.dag()

        H_2 = tensor([a_exp, b_exp.dag(), I_s, S_p])
        H_2 += H_2.dag()
        
        H_list += [H_1, H_2]
    
    H_free_phon = tensor( a.dag() * a, I_a, I_s, I_s) + 3**0.5 * tensor(I_a, a.dag() * a, I_s, I_s) -\
    delta_1 * tensor(I_a, I_a, S_z, I_s) - delta_2 * tensor(I_a, I_a, I_s, S_z)
    
    H_list += [H_free_phon]
    
    return H_list
def get_H_ions_phonons_2(system, mu):
    
    N_sys = system.N_sys_modes
    N_aux = system.N_aux
    dim_sys = system.dim_sys
    dim_aux = system.dim_aux

    if dim_sys != 2:
        raise ValueError("dim_sys != 2")

    if N_sys != N_aux:
        raise ValueError("N_sys != N_aux")
        
    if N_sys != 2:
        raise NotImplementedError
    
    I_a = identity(dim_aux)
    a = destroy(dim_aux)
    A = a + a.dag()
    
    I_s = identity(dim_sys)
    
    mu_c = mu / 2**0.5
    mu_r = mu * (4/3)**(1/4)
    
    H_1 = 2*mu_c* tensor([A, I_a, tensor(S_z, I_s)+tensor(I_s, S_z)]) +     mu_r * tensor([I_a, A, tensor(S_z, I_s)-tensor(I_s, S_z)])
    
    H_free_phon = tensor( a.dag() * a, I_a, I_s, I_s) + 3**0.5 * tensor(I_a, a.dag() * a, I_s, I_s)
    
    return H_1, H_free_phon
def get_H_ions_phonons_3(system, mu_abs):
    
    N_sys = system.N_sys_modes
    N_aux = system.N_aux
    dim_sys = system.dim_sys
    dim_aux = system.dim_aux

    if dim_sys != 2:
        raise ValueError("dim_sys != 2")

    if N_sys != N_aux:
        raise ValueError("N_sys != N_aux")
        
    if N_sys != 2:
        raise NotImplementedError
    
    I_s = identity(dim_sys)
    I_a = identity(dim_aux)
    a = destroy(dim_aux)
    A = a + a.dag()
    
    H_list = []
    for mu in [mu_abs,-mu_abs]:
        mu_c = mu / 2**0.5
        mu_r = mu * (4/3)**(1/4)

        a_exp = (1j * mu_c * A).expm()
        b_exp = (1j/2 * mu_r * A).expm()

        H_1 = tensor([a_exp, b_exp, S_p])
        H_1 += H_1.dag()

        H_2 = tensor([a_exp, b_exp.dag(), S_p])
        H_2 += H_2.dag()
        
        H_list += [H_1, H_2]
    
    H_free_phon = tensor( a.dag() * a, I_a, I_s, I_s) + 3**0.5 * tensor(I_a, a.dag() * a, I_s, I_s)
    H_free_1 = tensor(I_a, I_a, S_z, I_s)
    H_free_2 = tensor(I_a, I_a, I_s, S_z)
    
    H_free_list = [H_free_phon, H_free_1, H_free_2]
    H_1_p, H_2_p, H_1_m, H_2_m = H_list
    
    return [H_1_p,[0,1,2]], [H_2_p,[0,1,3]], [H_1_m,[0,1,2]], [H_2_m,[0,1,3]], H_free_list
def get_H_ions_phonons_4(system, mu_abs):
    
    N_sys = system.N_sys_modes
    N_aux = system.N_aux
    dim_sys = system.dim_sys
    dim_aux = system.dim_aux

    if dim_sys != 2:
        raise ValueError("dim_sys != 2")

    if N_sys != N_aux:
        raise ValueError("N_sys != N_aux")
        
    if N_sys != 2:
        raise NotImplementedError
    
    I_s = identity(dim_sys)
    I_a = identity(dim_aux)
    a = destroy(dim_aux)
    A = a + a.dag()
    
    H_list = []
    for mu in [mu_abs,-mu_abs]:
        mu_c = mu / 2**0.5
        mu_r = mu * (4/3)**(1/4)

        a_exp = (1j * mu_c * A).expm()
        b_exp = (1j/2 * mu_r * A).expm()

        H_1 = tensor([a_exp, b_exp, S_p])
        H_1 += H_1.dag()

        H_2 = tensor([a_exp, b_exp.dag(), S_p])
        H_2 += H_2.dag()
        
        H_list += [H_1, H_2]
    
    H_free_phon = tensor( a.dag() * a, I_a) + 3**0.5 * tensor(I_a, a.dag() * a)
    
    H_1_p, H_2_p, H_1_m, H_2_m = H_list
    
    return [H_1_p,[0,1,2]], [H_2_p,[0,1,3]], [H_1_m,[0,1,2]], [H_2_m,[0,1,3]], [H_free_phon, [0,1]], [S_z, [2]], [S_z, [3]]
def get_H_ions_phonons_5(system, mu_abs):
    
    N_sys = system.N_sys_modes
    N_aux = system.N_aux
    dim_sys = system.dim_sys
    dim_aux = system.dim_aux

    if dim_sys != 2:
        raise ValueError("dim_sys != 2")

    if N_sys != N_aux:
        raise ValueError("N_sys != N_aux")
        
    if N_sys != 2:
        raise NotImplementedError
    
    I_s = identity(dim_sys)
    I_a = identity(dim_aux)
    a = destroy(dim_aux)
    A = a + a.dag()
    
    H_list = []
    for mu in [mu_abs,-mu_abs]:
        mu_c = mu / 2**0.5
        mu_r = mu * (4/3)**(1/4)

        a_exp = (1j * mu_c * A).expm()
        b_exp = (1j/2 * mu_r * A).expm()

        H_1 = tensor([a_exp, b_exp, S_p, I_s])
        H_1 += H_1.dag()

        H_2 = tensor([a_exp, b_exp.dag(), I_s, S_p])
        H_2 += H_2.dag()
        
        H_list += [H_1, H_2]
    
    H_free_phon = tensor( a.dag() * a, I_a, I_s, I_s) + 3**0.5 * tensor(I_a, a.dag() * a, I_s, I_s)
    H_free_1 = tensor( I_a, I_a, S_z, I_s)
    H_free_2 =  tensor( I_a, I_a, I_s, S_z)
    
    H_1_p, H_2_p, H_1_m, H_2_m = H_list
    
    return H_1_p, H_2_p, H_1_m, H_2_m, H_free_phon, H_free_1, H_free_2
import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
def mkl_set_num_threads(cores):
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))
def create_copy_list(obj_list, N):
    obj_list_out = []
    for i in range(N):
        obj_list_out += deepcopy(obj_list)
    return obj_list_out
try:
    from my_ptrace import ptrace_legacy
except:
    from libs.my_ptrace import ptrace_legacy
def ptrace_mq(rho, sel):
    rho.data.sort_indices()
    res = ptrace_legacy(rho.data, [rho.dims]*2, rho.get_shape(), sel)
    return MyQobj(res[0], dims = res[1][0])
def sys_to_state_inds(sys_inds, aux_inds):

    state_inds = copy(sys_inds)
    for ind in aux_inds:
        state_inds = [m if m<ind else m+1 for m in state_inds]

    return state_inds
def is_slice_in_list(s,l):
    return set(s).issubset(set(l))
def map_to_sub2full_space(oper_in, full_space_dims, subspace_inds):
    '''
    Map operator from a subspace to an operator in a large space.

    Parameters:
    ----------
    oper_in: Qobj, 'oper'
                operator to convert.
                
    full_space_dims: List
                list of dims in full spase
                
    subspace_inds: [[mode_ind, lvls_list],...]
                Logical subspase. List of mode_ind of the state in the correponding order and the indices of levels, 
                lvls_list, of the mode with mode_ind corresponded to {|i>} in the correponding order.
    '''
    
    sub_mode_dims = get_q_dim_list(oper_in)
    if len(subspace_inds) != len(sub_mode_dims):
        raise ValueError('Dimensions mismatch')
    
    sub_mode_inds_list = [ind[0] for ind in subspace_inds]    
    sub_lvls_list = [ind[1] for ind in subspace_inds]    
    for i, lvls in enumerate(sub_lvls_list):
        if len(lvls) != sub_mode_dims[i]:
            raise ValueError('Dimensions mismatch')
     
    inds_mode_to_tens = []
    dims_mode_to_tens = []
    
    # Complete levels
    lvls_map_list = []
    for mode_ind, lvls_list in subspace_inds:
        lvls_map_list += [one_mode_sub2full_space( full_space_dims[mode_ind], lvls_list )]

    lvls_map = tensor(lvls_map_list)
    if oper_in.type is 'oper':
        oper_complete_lvls = lvls_map * oper_in * lvls_map.dag()
    elif oper_in.type is 'ket':
        oper_complete_lvls = lvls_map * oper_in
    elif oper_in.type is 'bra':
        oper_complete_lvls = oper_in * lvls_map.dag()
    else:
        raise NotImplementedError
    
    # Complete modes
    oper_out = make_multimode_H(oper_complete_lvls, sub_mode_inds_list, full_space_dims)
    
    return oper_out

def one_mode_sub2full_space( dim_full, inds_sub ):
    '''
    Returns map from which converts a single mode of a subspace to a mode of a large space. 
    '''
    
    dim_sub = len(inds_sub)
    
    if dim_sub==0:
        raise ValueError('len(inds_sub) has to be > 0')
        
    if max(inds_sub) > dim_full-1:
        raise ValueError('Index of the subsystem is larger then dim of the full system')
        
    projector = sum( fock(dim_full, ind) * fock(dim_sub, i).dag() for i, ind in enumerate(inds_sub))
    
    return projector
# def get_aux_proj_list(I_sys_list, dim_aux_list, aux_inds, order):
#     '''
#     Return list of projectors to the basis of the auxiliary subsystem.
#     '''   
#     N_aux = len(dim_aux_list)
    
#     basis_aux_list = get_basis_list(dim_aux_list)
    
#     proj_list = []
#     for basis in basis_aux_list:
        
#         state_list = [None]*N_aux + I_sys_list
#         state_list = permute_list(state_list, order)
#         for i, b in enumerate(basis):
#             state_list[aux_inds[i]] = fock(dim_aux_list[i],b)

#         proj_list += [tensor(state_list).dag()]
    
#     return proj_list

def get_aux_proj_list(system, inds_to_act):
    '''
    Return list of projectors to the basis of the auxiliary subsystem.
    '''   
    
    aux_inds = [i[0] for i in system.aux_mode_inds if i[0] in inds_to_act]
    dim_aux_list = [system.dims_state_list[i] for i in aux_inds]
    dim_sys_list = [system.dims_state_list[i] for i in inds_to_act if i not in aux_inds]
    dim_to_act_list = [system.dims_state_list[i] for i in inds_to_act]
    
    N_aux = len(aux_inds)
    I_list = [identity(system.dims_state_list[ind]) for ind in inds_to_act]
    
    basis_aux_list = get_basis_list(dim_aux_list)
    
    proj_list = []
    for basis in basis_aux_list:
        
        state_list = deepcopy(I_list)
        for i, b in enumerate(basis):
            state_list[aux_inds[i]] = fock(dim_aux_list[i],b)
        proj = tensor(state_list).dag()
#         proj.dims = [dim_sys_list, dim_to_act_list]
        
        proj_list += [proj]
    return proj_list
# def get_aux_lvls_proj_list(N_aux_lvls, dim_sys_list):
#     '''
#     Return list of projectors to the basis of the auxiliary subsystem.
#     '''
#     if dim_sys_list.count(dim_sys_list[0]) != len(dim_sys_list):
#         raise NotImplementedError("dim_sys_list are different : " + str(dim_sys_list))
    
#     dim_sys = dim_sys_list[0]
#     N_sys = len(dim_sys_list)
    
#     projs_1_list = one_mode_projs_aux_lvl( dim_sys + N_aux_lvls, N_aux_lvls )
    
#     proj_lists = [[]]
#     for i in range(N_sys):
#         proj_lists = [ proj_list + [projs_1] for proj_list in proj_lists for projs_1 in projs_1_list]
    
#     return [ tensor(proj_list) for proj_list in proj_lists ]
# def one_mode_projs_aux_lvl( N_dim, N_aux_lvls ):
#     '''
#     The aux level is the last one and its traced away.
#     '''
#     if N_aux_lvls > 1:
#         raise NotImplementedError
        
#     proj_0 = sum( fock(N_dim-1, i) * fock(N_dim, i).dag() for i in range(N_dim-1) )
#     proj_1 = fock(N_dim-1, 0)*fock(N_dim, N_dim-1).dag()
    
#     return [proj_0, proj_1]
def get_sum_Z(state):
    if state.type is 'ket':
        state = ket2dm(state)
    N = get_q_N_mode(state)
    ZZ = sum(make_multimode_oper(-sigmaz(), N, [i]) for i in range(N))
    return (state*ZZ).tr()

def is_eig_sum_Z(state):
    if state.type is 'ket':
        state = ket2dm(state)
    N = get_q_N_mode(state)
    ZZ = sum(make_multimode_oper(-sigmaz(), N, [i]) for i in range(N))
    
    Z_1 = (state*ZZ).tr()
    Z_2 = (state*ZZ**2).tr()
    
    D = Z_1**2-Z_2
    if abs(D)<=1e-8:
        return True
    else:
        return False
def check_CP(rho):
    N = get_q_N_mode(rho)
    sxx = tensor([S_x]*N)
    rho_2 = sxx*rho.permute(range(N)[::-1])*sxx
#     rho_2 = rho.permute(range(N)[::-1])
    return (rho_2 - rho).norm()

def check_P(rho):
    N = get_q_N_mode(rho)
    rho_2 = rho.permute(range(N)[::-1])
#     rho_2 = rho.permute(range(N)[::-1])
    a = (rho_2 - rho).norm()
    if abs(a)<1e-8:
        return True
    else: return False
def get_projector(N_qub, N_dim, n_cut):
    
    basis_dm = [ fock_dm(N_dim, i) for i in range(n_cut) ]
    project = sum( basis_dm )
    
    return tensor( [project] * N_qub )
def map_params(params, params_map):
    
    if params_map is None:
        return params
    
    N_params = sum(len(p) for p in params_map)
    
    params_out = np.zeros(N_params)
#     for i, map_inds in enumerate(params_map):
#         params_out[map_inds] = params[i]
        
    for i, map_inds in enumerate(params_map):
        for j in map_inds:
            if j == 0:
                params_out[abs(j)] = params[i]
            else:
                params_out[abs(j)] = np.sign(j) * params[i]
    
#     for i in range(1,len(params_out)//2+1):
# #         params_out[-i] = 2*pi-params_out[-i]
#         params_out[-i] = -params_out[-i]

    return params_out

def reverse_map(dU_list, params_map):
    
    if params_map is None:
        return dU_list
    
    if not isinstance(dU_list, np.ndarray):
        dU_list = np.array(dU_list)
    
    dU_list_out = [sum(dU_list[map_inds]) for map_inds in params_map]
    
#     dU_list_out = [0]*len(params_map)
#     for i, map_inds in enumerate(params_map):
#         for j in map_inds:
#             if j == 0:
#                 dU_list_out[i] += dU_list[abs(j)]
#             else:
#                 dU_list_out[i] += np.sign(j) * dU_list[abs(j)]
                
#     dU_list_out = [dU_list[map_inds[0]]-dU_list[map_inds[1]] for map_inds in params_map]
    
    return dU_list_out
def save_to_g(*args, is_stop = False):
    global g
    g = args
    if is_stop:
        sys.exit()
from qutip.cy.spconvert import arr_coo2fast, cy_index_permute
def my_permute(Q, order):
    if get_q_N_mode(Q)!=len(order) or get_q_N_mode(Q)!= np.max(order)+1:
        raise ValueError("len(Q)!=len(order) or len(Q)!= np.max(order)+1")
    Qcoo = Q.data.tocoo()
    
    cy_index_permute(Qcoo.row,
                     np.array(Q.dims[0], dtype=np.int32),
                     np.array(order, dtype=np.int32))
    cy_index_permute(Qcoo.col,
                     np.array(Q.dims[1], dtype=np.int32),
                     np.array(order, dtype=np.int32))

    new_dims = [[Q.dims[0][i] for i in order], [Q.dims[1][i] for i in order]]
    new_data = arr_coo2fast(Qcoo.data, Qcoo.row, Qcoo.col, Qcoo.shape[0], Qcoo.shape[1])
    
    return Qobj(new_data, dims = new_dims)

def my_permute_mq(Q, order):
    if list(range(len(order))) == list(order):
        return Q
    
    if len(Q.dims[0])!=len(order) or len(Q.dims[0])!= np.max(order)+1:
        raise ValueError("len(Q)!=len(order) or len(Q)!= np.max(order)+1")
    if Q.format is FORMAT_DENSE:
        Qcoo = sp.sparse.coo_matrix(Q.data)
    else:
        Qcoo = Q.data.tocoo()
    
    cy_index_permute(Qcoo.row,
                     np.array(Q.dims[0], dtype=np.int32),
                     np.array(order, dtype=np.int32))
    cy_index_permute(Qcoo.col,
                     np.array(Q.dims[1], dtype=np.int32),
                     np.array(order, dtype=np.int32))

    new_dims = [[Q.dims[0][i] for i in order], [Q.dims[1][i] for i in order]]
    new_data = arr_coo2fast(Qcoo.data, Qcoo.row, Qcoo.col, Qcoo.shape[0], Qcoo.shape[1])
    
    if Q.format is FORMAT_DENSE:
        new_data = new_data.toarray()
        
    return MyQobj(new_data, q_type = Q.q_type, dims = new_dims)
def super_tensor_with_pure_state(N_dim, psi_to_tens):
    I = identity(N_dim)
    I_psi = tensor(I, psi_to_tens).data
    return my_sprepost(I_psi, I_psi.T.conj()).toarray() 
def my_spre(A):
    return sp.sparse.kron(sp.identity(np.prod(A.shape[1])), A, format='csr')

def my_spost(A):
    return sp.sparse.kron(A.T, sp.identity(np.prod(A.shape[0])), format='csr')

def my_sprepost(A, B):
    return sp.sparse.kron(B.T, A, format='csr')


def my_spre_diag(A):
    return np.stack([A]*len(A), axis=0).flatten()

def my_spost_diag(A):
    return np.stack([A]*len(A), axis=1).flatten()


def to_super_oper(A):
    return my_sprepost(A, A.conj().T)

def to_super_H(A):
    return my_spre(A) - my_spost(A)

def to_super_oper_diag(A):
    return (A * A.reshape(-1,1)).flatten()

def to_super_H_diag(A):
    return my_spre_diag(A) - my_spost_diag(A)
def get_q_N_mode(state):
    if not isinstance(state, Qobj):
        raise ValueError("state has to be a Qobj")
        
    return len(state.dims[0])

def get_q_dim_list(state):
    if not isinstance(state, Qobj):
        raise ValueError("state has to be a Qobj")
    
    if state.type == 'bra':
        return state.dims[1]
    elif len(state.dims[0]) == 0:
        return 0
    else:
        return state.dims[0]
def my_op_to_vec(rho):
    return rho.T.reshape(-1,1)
    
def my_vec_to_op(vec):
    
    dim_out = vec.shape[0]**0.5
    if dim_out%1 != 0:
        raise ValueError
    else:
        dim_out = int(dim_out)
    
    op = vec.reshape(dim_out, -1).T
    
    if not op.flags['C_CONTIGUOUS']:
        op = np.ascontiguousarray(op)
        
    return op
def kron_all(list_to_kron):
    tens = list_to_kron[0]
    for oper in list_to_kron[1:]:
        tens = np.kron(tens, oper)
        
    return tens
def exp_mult_real_imag(A, B_real, B_imag, t=1):
        
    if B_real is 0:
        B = B_imag
    elif B_imag is 0:
        B = B_real
    else:
        B = np.append(B_real,B_imag, axis=1)
    C, S, _,_,_,_,_ = funmv(t,A,B,flag = 'cos.sin',parallel = True)

    
    if B_real is 0:
        C1 = 0
        S1 = 0
        
        C2 = C
        S2 = S
    elif B_imag is 0:
        C1 = C
        S1 = S
        
        C2 = 0
        S2 = 0
    else:
        n = B_real.shape[1]
        C1 = C[:,:n]
        S1 = S[:,:n]
        
        C2 = C[:,n:]
        S2 = S[:,n:]
        
    B_real_out = C1 + S2
    B_imag_out = C2 - S1
    
    return B_real_out, B_imag_out
def exp_S3_x(t0):
    '''
    exp(-i*t*sigma_x/2**0.5) for qutrit
    1/2**0.5 is a coefficient in the Hamiltonian
    '''
    t = t0/2**0.5
    c = math.cos(t)
    
    A = -1j*math.sin(t)*2**-0.5
    
    B = 0.5*(1+c)
    C = 0.5*(-1+c)
    
    
    return np.array([[B,A,C],[A,c,A],[C,A,B]])

def exp_S2_x(t0):
    '''
    exp(-i*t*sigma_x/2) for qubit
    1/2 is a coefficient in the Hamiltonian
    '''
    t = t0/2
    a = math.cos(t)
    b = -1j*math.sin(t)
    
    return np.array([[a,b],[b,a]])
    
    return np.array([[a,b],[b,a]])
class Cache(object):     
    """
    Object to cache objects to RAM and retreave them by "meta"
    """
    stored = []
    
    def __init__(self):
        
        self.stored = []
    
    def add_to_cache( self, obj, meta ) :
        
        self.stored += [( obj, meta )]
        
    def get_from_cache( self, meta ) :
        
        return next( ( s[0] for s in self.stored if s[1] == meta), None )
        
    def clear_cache( self ) :
        
        self.stored = []

global_cache = Cache()
def norm_array(y):
    return (y - np.min(y))/(np.max(y) - np.min(y))
def purity(rho):
    if isinstance(rho,MyQobj):
        return (rho*rho).tr()
    if isinstance(rho,Qobj):
        return (rho**2).tr()
    if isinstance(rho,np.ndarray):
        return np.real((np.dot(rho,rho)).trace())
def E_rel(E_array, opt_props):
    return (E_array - opt_props.targ_E)/(opt_props.E_max - opt_props.targ_E)
def is_diag(H):
    return np.all(H.data == np.diag(H.diag()))
def pick_params(params, ind, n):
    if ind + n > len(params):
        raise ValueError('ind + n > len(params)')
        
    return params[ind:ind + n], ind + n
def make_multimode_oper(oper_in, N_mode, inds_array, dim = 2):
    '''
    Create an 'N_mode' operator with 'oper_in' acting to modes in 'inds_array' in qubit basis
    '''
    if isinstance(oper_in, Qobj):
        opers_list = [qeye(dim)] * N_mode
        for ii in range(len(inds_array)):
            opers_list[inds_array[ii]] = oper_in
        return tensor(opers_list)
    else:
        opers_list = [np.identity(dim)] * N_mode
        for ii in range(len(inds_array)):
            opers_list[inds_array[ii]] = oper_in
        
        tens = opers_list[0]
        for oper in opers_list[1:]:
            tens = np.kron(tens, oper)
        return tens
def permute_list(lst, inds):
    
    return [lst[i] for i in inds]
def get_basis_list(dim_list):
    '''
    Return array qubit basis for dims in dim_list. For [2,2] return an array: [[0,0],[1,0],[0,1],[1,1]].
    '''
    N = len(dim_list)
    if N == 0:
        return [[]]

    basis_list = [[i] for i in range(dim_list[0])]
    for i in range(N-1):
        basis_list = [ basis_el + [j] for basis_el in basis_list for j in range(dim_list[i+1])]
    
    return basis_list
def cut_params(params, bounds):
    
    if not isinstance(params, collections.Iterable):
        params = [params]
    
    if len(bounds) == 1:
        bounds = bounds * len(params)
    elif len(bounds) != len(params):
        raise ValueError("len(bounds) != len(params)")
    
    params_out = []
    for i, p in enumerate(params):
        min_val, max_val = bounds[i]
        d = max_val - min_val
        
        if min_val != -np.inf and max_val != np.inf:
            p = p - min_val
            N2 = int(p *1./ (2*d))
            p2 = p - N2 * 2*d
            N1 = int(p2 *1./ d)
            p1 = p2 - N1 * d
            p_out = min_val + (1-abs(N1)) * abs(p1) + abs(N1) * (d - abs(p1))
        elif min_val != np.inf and max_val == np.inf:
            if p < min_val : p_out = min_val-(p - min_val)
            else: p_out = p
        elif min_val == -np.inf and max_val != np.inf:
            if p > max_val : p_out = (max_val - p) - max_val
            else: p_out = p
            
        params_out += [p_out]
        
    return np.array(params_out)
def consistent_opt(simulator, depth_list, layer, params_init, params_add, delete = []):
    '''
    Optimize continuously increasing depth in depth_list and reusing the optimal parameters.
    '''
    
    F_array_list = []
    F_st_list = []
    nfev_list = []
    params_list = []
    result_list = []
    for i, N_layers in enumerate(sorted(depth_list)):

        gates = layer * N_layers
        for j in delete:
            del gates[j]
        
        kraus_gen_list = [ KrausGenerator( gates, simulator.sys_props )]
        simulator.set_kraus_gen(kraus_gen_list)
        
        if params_add is not None:
            if i>0:
                if np.isscalar(params_add):
                    to_add = [params_add]*(simulator.N_params - len(params_init))
                else:
                    to_add = params_add * (depth_list[i] - depth_list[i-1])

                params_init = list(params_init) + to_add
            simulator.set_params_init(params_init)
        else:
            if isinstance(params_init, numbers.Number):
                simulator.set_params_init(params_init)
            else:
                simulator.set_params_init(params_init[i])
        
        print("N_layers = " + str(N_layers))
        result = optimize_params( simulator )
        result.gen_eigen_props()
        result_list += [result]
        
        sim_res = simulator.simulation( result.res.x, return_F = True, step_max = simulator.opt_props.N_iter*2 )
        F_array = sim_res.F_array
        F_array_list += [F_array]
        F_st_list += [result.F_st_an]
        nfev_list += [result.res.nfev]
        params_list += [result.res.x]
        
        if params_add is not None:
            params_init = result.res.x
        
        print 
        print("F_st = " + str(result.F_st_an) + ", nfev = " + str(result.res.nfev))
        print 
        
    nfev_tot = sum(nfev_list)
    return result_list, F_st_list, F_array_list
# def f_approx(x, a, b, c):
#     '''
#     approximate values in 'x'
#     '''
#     c = round(c)
#     H = np.heaviside(x - c,b)
#     return b *( (1 - H)  + H * np.exp(- a * (x - c)))

def f_approx( x, a, b):
    '''
    approximate values in 'x'
    '''
    
    y2 = b * np.exp(- a * x)
    y1 = 1 - y2
    
    return y2

def optimiz_func_1D( x, ind, params, simulator):
    params[ind] = x
    
    return optimiz_func( params, simulator)
    
def optimiz_func( params, simulator, cost_func):
    '''
    cost function based on energy at the 'ind_step'-th dissipative cycle and 'prob_array' up to 'N_samp'-th cycle.
    '''    
    global params_g
    params_g = params
    
    if isinstance(simulator,list):
        simulator_list = simulator
        simulator = simulator_list[0]
    else:
        simulator_list = [simulator]
    
    opt_props = simulator.opt_props
    system = simulator.system

    grad_en_cost = 0
    en_cost = 0
    # results of simulation
    for i, s in enumerate(simulator_list):
        sim_res = s.simulation(params)
#         sim_res = s.simulation_st_st(params)
    
        grad_en_cost += sim_res.jac
        en_cost += sim_res.E
#         en_cost -= sim_res.P_array[i][0]
        rho_cost = sim_res.rho_out 
    
    if opt_props.use_probs:
        raise NotImplementedError
#         opt_var_sum = np.sum( opt[1] )
#         cost = -1e-2 * (a_sum ) + en_cost + 1e2* opt_var_sum
#         cost = -1e1 * (sim_res.a_sum ) + en_cost
    else:
        a_sum = 0
        opt_var_sum = 0
#         cost = en_cost
    
    if en_cost < simulator.MIN_glob: simulator.MIN_glob = en_cost
    if opt_props.print_data:
        E_min = sum(simulator.model.E_min for simulator in simulator_list) if simulator_list[0].model.E_min is not None else 'N/a'
        printNum(en_cost, simulator.MIN_glob, E_min, file_logs = opt_props.file_logs)
        
    cost = cost_func([en_cost, params])
    if opt_props.jac:
#         n_p = len(params)
#         d_params = params.reshape(1,n_p)
#         d_params = np.repeat(d_params, n_p) + eps * np.identity(n_p)
#         grad_cost = [ cost_func([grad_en_cost[i], d_params[i]]) for i in range(n_p)]
        dC_dE = (cost_func([en_cost+eps, params]) - cost)/eps
        grad_cost = []
        for i in range(len(params)):
            d_params = copy(params)
            d_params[i] += eps
            dC_dP = (cost_func([en_cost, d_params]) - cost)/eps
            grad_cost += [dC_dE * grad_en_cost[i] + dC_dP]
#         print(grad_en_cost)
#         print(grad_cost)
#         print()
        return cost, np.array(grad_cost)
    else:
        return cost

def optimize_params( simulator, cost_func = lambda args: args[0] ):
    '''
    Parameters
    ----------
    cost_func : function of args = [energy, parameters]
        cost function.
    '''
    if isinstance(simulator,list):
        simulator_list = simulator
        simulator = simulator_list[0]
    else:
        simulator_list = [simulator]
    global_cache.clear_cache()
    for s in simulator_list:
        s.MIN_glob = np.inf
    opt_props = simulator.opt_props
    system = simulator.system
    model = simulator.model
    
    if model.E_max is None:
        tol_abs = opt_props.tol_rel
    else:
        tol_abs = opt_props.tol_rel * (model.E_max - model.E_min)
        
    # optimization
    if opt_props.method == "basinhop":
        optimiz_func_2 = functools.partial(optimiz_func, simulator = simulator_list, cost_func = cost_func)
        minimizer_kwargs = {"method": "BFGS", "tol": tol_abs, 'jac':opt_props.jac}
        res = sp.optimize.basinhopping(optimiz_func_2, simulator.params_init, 
                                       niter = opt_props.maxiter, T = 1, minimizer_kwargs = minimizer_kwargs, seed = 123123)
    
    elif opt_props.method == "brute":
        optimiz_func_2 = functools.partial(optimiz_func, simulator = simulator_list, cost_func = cost_func)
        res = minimize(optimiz_func_2, bounds = opt_props.ranges)
        
    elif opt_props.method == '1D_consist':
        params = opt_props.params_init
        for i in range(1):
            for ind in range(opt_props.N_params):
                options = {"maxiter" : opt_props.maxiter }
                res = sp.optimize.minimize(optimiz_func_1D, params[ind], args = (ind, params, opt_props), 
                                           method = "BFGS", tol = opt_props.tol, options = options)
                
                params[ind] = res.x
        res.x = params
        
    else:
        options = {"maxiter" : opt_props.maxiter }
        res = sp.optimize.minimize(optimiz_func, simulator.params_init, args = (simulator_list, cost_func), 
                                   method = opt_props.method, tol =tol_abs, options = options, callback = callback, jac = opt_props.jac)

    
#     res = scipydirect.minimize(optimiz_func_2, bounds=bounds, maxf=200000)    
#     return res

#     bounds = [(0, 20)] * opt_props.N_params
#     res = sp.optimize.differential_evolution(optimiz_func_2, bounds, 
#                                           maxiter = opt_props.maxiter, tol = opt_props.tol)

#     res = scipy.optimize.brute(optimiz_func_2, bounds)
    
    return res
def callback(xk):
    global_cache.clear_cache()

