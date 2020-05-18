import warnings
import collections

from qutip import*
import numpy as np
import scipy as sp
from copy import copy
from scipy.sparse.linalg import expm_multiply
from math import *
import matplotlib.pyplot as plt
import math
import numbers
import sys
import numpy, scipy.io

import functools
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm
from scipy.integrate import ode
from qutip.solver import Options

import scipy.io as sio
from scipy.sparse.linalg import LinearOperator, aslinearoperator
def funmv(t,A,b,flag,M = None, prec = None, shift = True, bal = False, full_term = False, parallel = True):
    
# %FUNMV   trigonometric and hyperbolic matrix functions times vectors
# %   [C,S,s,m,mv,mvd] = FUNMV(t,A,B,FLAG,[],PREC) computes f(t*A)*B or
# %   f(t*SQRTM(A))B, where f is a trigonometric or hyperbolic matrix function
# %   and SQRTM denotes any matrix square root, without
# %   explicitly forming f(t*A) or SQRTM(A). PREC is the required accuracy,
# %   'double', 'single' or 'half', and defaults to CLASS(A).
# %   A total of mv products with A or A^* are used, of which mvd are
# %   for norm estimation.
# %   The full syntax is
# %   [C,S,s,m,mv,mvd,unA] = funmv(t,A,b,flag,M,prec,shift,bal,full_term).
# %   unA = 1 if the alpha_p were used instead of norm(A).
# %   If repeated invocation of FUNMV is required for several values of t
# %   or B, it is recommended to provide M as an external parameter as
# %   M = SELECT_TAYLOR_DEGREE_TRIG(A,b,flag,m_max,p_max,prec,shift,bal,true).
# %   This also allows choosing different m_max and p_max.

# %   The outputs C and S depend on the input FLAG as follows:

# %   FLAG = 'cos.sin'        , C = cos(tA)B  and S = sin(tA)B
# %   FLAG = 'cosh.sinh'      , C = cosh(tA)B and S = sinh(tA)B
# %   FLAG = 'cos.sinc'       , C = cos(tA)B  and S = sinc(tA)B
# %   FLAG = 'cosh.sinch'     , C = cosh(tA)B and S = sinch(tA)B
# %   FLAG = 'cos.sinc.sqrt'  , C = cos(t*sqrt(A))B and S = sinc(t*sqrt(A))B
# %   FLAG = 'cosh.sinch.sqrt', C = cosh(t*sqrt(A))B and S = sinch(t*sqrt(A))B

# %   The parameter SHIFT is optinal only if FLAG = 'cos.sin'.

# %   Example: 
# %   To evaluate a combination of the form 
# %                   cos(t*sqrt(A))*y0 + t*sinc(t*sqrt(A))*y1,
# %   it is C(:,1) + t*S(:,2), where [C,S] = funmv(t,A,[y0,y1],'cos.sinc.sqrt') 
# %   

# %   Reference: A. H. Al-Mohy, A New Algorithm for Computing the Actions 
# %   of Trigonometric and Hyperbolic Matrix Functions

# %   Awad H. Al-Mohy, January 18, 2018.

    if flag not in ['cos.sin', 'cosh.sinh', 'cos.sinc', 'cosh.sinch', 'cos.sinc.sqrt', 'cosh.sinch.sqrt']:
        raise ValueError('funmv:NoInput', 'Choose a correct input of the parameter FLAG.')


    flag1 = flag == 'cos.sin'
    flag2 = flag == 'cosh.sinh'
    if flag1: flag, sign = 1, 1
    if flag2: flag, sign = 1, 0
    if flag == 'cos.sinc': flag, sign = 1, 1
    if flag == 'cosh.sinch': flag, sign = 1, 0
    if flag == 'cos.sinc.sqrt': flag, sign = 0, 1
    if flag == 'cosh.sinch.sqrt': flag, sign = 0, 0

    if not (flag1 or flag2): shift = False
    if bal:
        D, B = balance(A)
        if _exact_1_norm(B) < _exact_1_norm(A): A, b = B, np.linalg.inv(A)*b
        else: bal = False

    n, n0 = A.shape[0], b.shape[1]
    pp = n0
    if flag: pp = 2*n0 #the number of matrix-vector prod., A(AB).  

    if shift and (flag1 or flag2):
        mu = _trace(A)/n
        tmu = t*mu
        A = A - mu*sp.sparse.identity(n)     

    if prec is None: prec = A.dtype.name
    t2 = t
    if not flag: t2 = t**2
    if M is None:
        tt = 1
        M,mvd,_,unA = select_taylor_deg_trig(t2*A,b,flag,prec)
        mv = mvd
    else:
        tt = t
        mv = 0
        mvd = 0
        unA = 0

    if prec == 'float64': tol = 2**(-53)
    elif prec == 'float32': tol = 2**(-24)
    elif prec == 'float16': tol = 2**(-10)
    
    s = 1
    if t == 0:
        m = 0
    else:
        m_max, p = M.shape
        S = np.diag(range(1,m_max+1))
        C = abs(tt)*M
        C = np.ceil(C.real)
        C = np.dot(C.T.conj(), S)
        C[C==0] = np.inf
        if p > 1:
            cost = np.min(np.min(C,0))
            m = (np.argmin(np.min(C,0))+1)#cost is the overall cost.
        else:
            cost, m = np.min(C,0), np.argmin(C,0)+1  #when C is one column. Happens if p_max = 2.
        
        if cost is np.inf: cost = 0
        s = int(np.max([cost*1./m, 1])  )
        
        
    undo_inside = False 
    undo_outside = False
    # undo shifting inside or outside the loop
    if shift:
        if flag1 and not np.isreal(tmu):
            cosmu = np.cos(tmu/s)
            sinmu = np.sin(tmu/s)
            undo_inside = True
        elif flag1 and np.isreal(tmu) and abs(tmu)> 0:
            cosmu = np.cos(tmu)
            sinmu = np.sin(tmu)
            undo_outside = True
        elif flag2 and abs(np.real(tmu)) > 0:
            cosmu = np.cosh(tmu/s)
            sinmu = np.sinh(tmu/s)
            undo_inside = True
        elif flag2 and not np.real(tmu) and abs(tmu)> 0:
            cosmu = np.cosh(tmu)
            sinmu = np.sinh(tmu)
            undo_outside = True

    mods = s%2
    C0 = np.zeros([n,n0])
    if mods: C0 = b/2.
    S = C0
    C1 = b
    for i in range(1,s+2):
        if i == s+1:
            S = 2*S
            C1 = S

        V = C1
        if undo_inside: Z = C1
        b = C1
        c1 = _exact_inf_norm(b)
        for k in range(1,m+1):            
            even = 2*k
            if i <= s:
                odd = even-1
                q = 1/(even+1)
            else:               # when i = s+1, compute Taylor poly. of sinc
                odd = even+1
                q = odd
                
            if parallel:
                if flag: b = mkl_dspmm(A, b)
                b = mkl_dspmm(A, b, alpha=(t/s)**2/(even*odd))
            else:
                if flag: b = A*b
                b = (A*b)*((t/s)**2/(even*odd))
                
            mv = mv + pp
            V = V + ((-1)**(k*sign))*b
            if undo_inside: Z = Z + (((-1)**(k*sign))*q)*b
                    

            c2 = _exact_inf_norm(b)
            if not full_term:
                if c1 + c2 <= tol*_exact_inf_norm(V):
                    break
                c1 = c2

        if undo_inside:
            if i <= s:
                V = V*cosmu + A*(Z*(((-1)**sign)*t*sinmu/s))
                mv = mv + n0
            else :        
                V = A*(V*(t*cosmu/s)) + Z*sinmu
                mv = mv + n0

        if i == 1:
            C2 = V
        elif i <= s:
            C2 = 2*V - C0

        if i <= s-1 and ( mods^i%2 ):
            S = S + C2   # sum of C_i for even i's if s is odd and vice versa     

        C0 = C1
        C1 = C2
    C = C2
    if undo_inside:
        S = V
    elif flag1 or flag2:
        S = A*(V*(t/s)) 
        mv = mv + n0
    else:
        S = V/s   # sinc(tA)B (or sinch(tA)B ) if flag = 1 and
                   # sinc(t*sqrt(A))B (or sinch(t*sqrt(A))B) if flag = 0

    if undo_outside:
        C = cosmu*C + (((-1)**sign)*sinmu)*S
        S = sinmu*C2 + cosmu*S

    if bal: 
        C = D*C
        S = D*S
    
    return C,S,s,m,mv,mvd,unA
# theta_d = sio.loadmat('/home/slava/Simulator/libs/theta_taylor_trig_double.mat')['theta']
# theta_s = sio.loadmat('/home/slava/Simulator/libs/theta_taylor_trig_single.mat')['theta']
# theta_h = sio.loadmat('/home/slava/Simulator/libs/theta_taylor_trig_half.mat')['theta']

def select_taylor_deg_trig(A,b,flag,prec = None,shift = False,bal = False,force_estm = False, m_max = 25,p_max = 5):
    
# %SELECT_TAYLOR_DEGREE_TRIG   Select degree of Taylor approximation.
# %   [M,MV,alpha,unA] = SELECT_TAYLOR_DEGREE_TRIG(A,m_max,p_max) forms a matrix M
# %   for use in determining the truncated Taylor series degree in FUNMV
# %   based on parameters m_max and p_max.
# %   MV is the number of matrix-vector products with A or A^* computed.

# %   Reference: A. H. Al-Mohy, A New Algorithm for Computing the Action 
# %   of Trigonometric and Hyperbolic Matrix Functions

# %   Awad H. Al-Mohy, August 08, 2017.

    if p_max < 2 or m_max > 25 or m_max < p_max*(p_max - 1):
        raise ValueError('>>> Invalid p_max or m_max.')

    n = A.shape[0]
    sigma = 1
    if not flag: sigma = 1./2
    if bal:
        D, B = balance(A)
        if _exact_1_norm(B) < _exact_1_norm(A): A = B

    if prec is None: prec = A.dtype.name
    if prec == 'float64':
        theta = theta_d
    elif prec == 'float32':
        theta = theta_s
    elif prec == 'float16':
        theta = theta_h
    
    if shift:
        mu = _trace(A)/n

        A = A - mu*sp.sparse.identity(n)

    mv = 0
    bound_hold = False
    normA = _exact_1_norm(A)
    ell = 2
    bound = 2*ell*p_max*(p_max+3)/(m_max*b.shape[1]) - 1
    c = normA**sigma
    if not force_estm and c <= theta[m_max]*bound:
        # Base choice of m on normA, not the alpha_p.
        unA = 0
        bound_hold = True 
        alpha = c*np.ones([p_max-1,1])
    if not force_estm and flag and not bound_hold:
        c, cost_d2 = normAm(A,2, parallel)
        c = c**(1./2)
        mv = mv + cost_d2
        if c <=  theta[m_max]*(bound - cost_d2):
            unA = 1.
            bound_hold = True
            alpha = c*np.ones([p_max-1,1])
            
    if not bound_hold:
        eta = np.zeros([p_max,1])
        alpha = np.zeros([p_max-1,1])
        for p in range(p_max):
            c, k = normAm(A,2*sigma*(p+2), parallel)
            c = c**(1./(2*p+4.))
            mv = mv + k
            eta[p] = c

        for p in range(p_max-1):
            alpha[p] = np.max([eta[p],eta[p+1]])
            
        unA = 2.
        
    M = np.zeros([m_max,p_max-1])
    for p in range(2,p_max+1):
        for m in range(p*(p-1)-1,m_max+1):
            M[m-1,p-2] = alpha[p-2]/theta[m-1]
            
    return M,mv,alpha,unA
def normAm(A,m, parallel):
# %NORMAM   Estimate of 1-norm of power of matrix.
# %   NORMAM(A,m) estimates norm(A^m,1).
# %   If A has nonnegative elements the estimate is exact.
# %   [C,MV] = NORMAM(A,m) returns the estimate C and the number MV of
# %   matrix-vector products computed involving A or A^*.

# %   Reference: A. H. Al-Mohy and N. J. Higham, A New Scaling and Squaring
# %   Algorithm for the Matrix Exponential, SIAM J. Matrix Anal. Appl. 31(3):
# %   970-989, 2009.

# %   Awad H. Al-Mohy and Nicholas J. Higham, September 7, 2010.
#     return _onenormest_matrix_power(A, m), m




    t = 1 # Number of columns used by NORMEST1.

    n = A.shape[0]
    if np.all(A.data>0):
        e = np.ones([n,1])
        for j in range(m):         # for positive matrices only
            e = A.conj().T*e

        c = _exact_inf_norm(e)
        mv = m
    else:
        parallel = False
        if not parallel:
            AT = A.T
            
        def mv(v):
            if parallel:
                for i in range(m):
                    v = mkl_dspmv(A,v)
            else:
                for i in range(m):
                    v = A*v
                    
            return v

        def mv_T(v):
            if parallel:
                for i in range(m):
                    v = mkl_dspmv(A,v, is_trnsa=True)
            else:
                for i in range(m):
                    v = AT*v
                    
            return v

        lin_op_A = LinearOperator(A.shape, matvec=mv, rmatvec = mv_T)

        c = sp.sparse.linalg.onenormest(lin_op_A,t)
#         c = sp.sparse.linalg.onenormest(aslinearoperator(A)**m,t)
#         mv = it[1]*t*m
        mv = m
    return c, mv
def _exact_inf_norm(A):
    # A compatibility function which should eventually disappear.
    if scipy.sparse.isspmatrix(A):
        return max(abs(A).sum(axis=1).flat)
    else:
        return np.linalg.norm(A, np.inf)


def _exact_1_norm(A):
    # A compatibility function which should eventually disappear.
    if scipy.sparse.isspmatrix(A):
        return max(abs(A).sum(axis=0).flat)
    else:
        return np.linalg.norm(A, 1)


def _trace(A):
    # A compatibility function which should eventually disappear.
    if scipy.sparse.isspmatrix(A):
        return A.diagonal().sum()
    else:
        return np.trace(A)
def expm_multiply_parallel(A, B, t=1.0, balance=False):
    
    if not B.flags["C_CONTIGUOUS"]:
        B = np.copy(B,order='C')
        
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError('the matrices A and B have incompatible shapes')
        
    ident = _ident_like(A)
    n = A.shape[0]
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')
    u_d = 2**-53
    tol = u_d
    mu = _trace(A) / float(n)
    A_2 = A - mu * ident
    A_1_norm = _exact_1_norm(A_2)
    
    if t*A_1_norm == 0:
        m_star, s = 0, 1
    else:
        ell = 2
        norm_info = LazyOperatorNormInfo(t*A, A_1_norm=t*A_1_norm, ell=ell)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
        
    return _my_expm_multiply_simple_core(A, B, t, mu, m_star, s, tol, balance)
def _my_expm_multiply_simple_core(A, B, t, mu, m_star, s, tol=None, balance=False):
    """
    A helper function.
    """
    if balance:
        raise NotImplementedError
    if tol is None:
        u_d = 2 ** -53
        tol = u_d
    F = B
    eta = np.exp(t*mu / float(s))
    eta = 1
    for i in range(s):
        c1 = _exact_inf_norm(B)
        for j in range(m_star):
            coeff = t / float(s*(j+1))
            B = mkl_zspmm(A, B, coeff)
            c2 = _exact_inf_norm(B)
            F = F + B
            if c1 + c2 <= tol * _exact_inf_norm(F):
                break
            c1 = c2
        F = eta * F
        B = F
#         print(i,j)
    return F
from scipy.sparse.linalg._expm_multiply import LazyOperatorNormInfo, _exact_inf_norm, _exact_1_norm, _trace, _ident_like, _fragment_3_1
_theta = {
        # The first 30 values are from table A.3 of Computing Matrix Functions.
        1: 2.29e-16,
        2: 2.58e-8,
        3: 1.39e-5,
        4: 3.40e-4,
        5: 2.40e-3,
        6: 9.07e-3,
        7: 2.38e-2,
        8: 5.00e-2,
        9: 8.96e-2,
        10: 1.44e-1,
        # 11
        11: 2.14e-1,
        12: 3.00e-1,
        13: 4.00e-1,
        14: 5.14e-1,
        15: 6.41e-1,
        16: 7.81e-1,
        17: 9.31e-1,
        18: 1.09,
        19: 1.26,
        20: 1.44,
        # 21
        21: 1.62,
        22: 1.82,
        23: 2.01,
        24: 2.22,
        25: 2.43,
        26: 2.64,
        27: 2.86,
        28: 3.08,
        29: 3.31,
        30: 3.54,
        # The rest are from table 3.1 of
        # Computing the Action of the Matrix Exponential.
        35: 4.7,
        40: 6.0,
        45: 7.2,
        50: 8.5,
        55: 9.9,
        }
def expL_multiply(rho_list, c_H_sum, c_stack, t_c=1.0):
    
    if not isinstance(rho_list,list):
        rho_list = [rho_list]
        
    if not rho_list[0].flags["C_CONTIGUOUS"]:
        rho_list = [np.copy(rho,order='C') for rho in rho_list]
#     if len(H.shape) != 2 or H.shape[0] != H.shape[1]:
#         raise ValueError('expected H to be like a square matrix')
#     if H.shape[1] != rho.shape[0]:
#         raise ValueError('the matrices H and rho have incompatible shapes')
    
#     if c_dag_list is None:
#         c_dag_list = [c.conj().T for c in c_list]
    H = c_H_sum
    ident = _ident_like(H)
    n = H.shape[0]
    rho = rho_list[0]
    if len(rho.shape) == 1:
        n0 = 1
    elif len(rho.shape) == 2:
        n0 = rho.shape[1]
    else:
        raise ValueError('expected rho to be like a matrix or a vector')
    u_d = 2**-53
    tol = u_d
    mu_H = _trace(H) / float(n)
    H_2 = H - mu_H * ident
    op_list = [H_2]
    
#     if c_dag_c_sum is not 0:
#         mu_cdc = _trace(c_dag_c_sum) / float(n)
#         c_dag_c_sum_2 = c_dag_c_sum - mu_cdc * ident
#         op_list += [c_dag_c_sum_2]

    norm_list = [_exact_1_norm(c) for c in op_list]
    t_list = [1, t_c]
    
    m_star_max, s_max = 0, 1
    for i, h in enumerate(norm_list):
        t = t_list[i]
        if t*h == 0:
            m_star, s = 0, 1
        else:
            ell = 2
            norm_info = LazyOperatorNormInfo(t*op_list[i], A_1_norm=t*h, ell=ell)
            m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)

        m_star_max, s_max = max(m_star_max, m_star), max(s_max, s)
    m_star_max, s_max = m_star_g, s_g
    rho_out_list = [_expL_multiply_simple_core(rho, c_H_sum, c_stack, t_c, m_star_max, s_max, tol)
                   for rho in rho_list]
    
    if len(rho_out_list)>1:
        return rho_out_list
    else:
        return rho_out_list[0]
def _expL_multiply_simple_core(rho, c_H_sum, c_stack, t_c, m_star, s, tol):
    """
    A helper function.
    """

    if tol is None:
        u_d = 2 ** -53
        tol = u_d
    F = rho
#     eta = np.exp(-t*sum(mu_list) / float(s))
    eta = 1
    for i in range(s):
        c1 = _exact_inf_norm(rho)
        for j in range(m_star):
            coeff = 1 / float(s*(j+1))
#             rho = -1j*(H*rho - rho*H) - 0.5*(c_dag_c_sum*rho + rho*c_dag_c_sum) + \
#             sum( c_list[k]*rho*c_dag_list[k] for k in range(len(c_list)))

            rho = cy_expL_multiply(rho, c_stack, c_H_sum, coeff )
            rho = np.triu(rho) + np.triu(rho,1).conj().T

            c2 = _exact_inf_norm(rho)
            F = F + rho
            if c1 + c2 <= tol * _exact_inf_norm(F):
                break
            c1 = c2
        F = eta * F
        rho = F
#         print(i,j)
    return F
def lind_iter(rho, H, c_dag_c_sum, c_list, c_dag_list, t_c):
    
    h = -1j*H
    if c_dag_c_sum is not 0:
        h += - 0.5*c_dag_c_sum
    
    rho_out = h*rho
    rho_out += rho_out.conj().T
    
    if len(c_list)>0:
        rho_out += sum( t_c*c_list[k]*rho*c_dag_list[k] for k in range(len(c_list)))
    return rho_out
def lind_iter_parallel(rho, H, c_dag_c_sum, c_list, coeff, t_c):
    
    h = -1j*H
    if c_dag_c_sum is not 0:
        h += - 0.5*c_dag_c_sum
    
    rho_out = mkl_zspmm(h, rho, coeff)
    rho_out += rho_out.conj().T
    
    if len(c_list)>0:
        rho_c_dag_list = [ mkl_zspmm(c, rho).conj().T for c in c_list]
        rho_c_dag_list = [np.copy(c,order='c') for c in rho_c_dag_list]
        rho_out += sum( mkl_zspmm(c_list[i], rho_c_dag_list[i], coeff*t_c) for i in range(len(c_list)) )
    return rho_out
def cy_expL_multiply(rho_in, c_stack, c_H_sum, coeff ):
    N = c_H_sum.shape[0]
    rho_out = np.zeros([N,N])*(1.+0.j)
    
    cy_mkl_z_syprd.expL_multiply( c_stack.data, c_stack.indptr, c_stack.indices, 
                                 c_H_sum.data, c_H_sum.indptr, c_H_sum.indices, 
                                 N, rho_in, coeff, rho_out )
    return rho_out
def precision(x1,x2):
    return numpy.linalg.norm(x1-x2)/ numpy.linalg.norm(x1)
def super_map_to_basis_mat(super_state):
    shape = super_state.shape
    dim = int(shape[0]**0.5)
    basis_vec_list = [super_state[:,i] for i in range(shape[1])]
    return [ b.reshape([dim,dim]).T for b in basis_vec_list]

def basis_mat_to_super_map(basis_list):
    basis_vec_list = [b.ravel('F') for b in basis_list]
    return np.vstack(basis_vec_list).T
def my_spre(A):
    return sp.sparse.kron(sp.identity(np.prod(A.shape[1])), A, format='csr')

def my_spost(A):
    return sp.sparse.kron(A.T, sp.identity(np.prod(A.shape[0])), format='csr')

def my_sprepost(A, B):
    return sp.sparse.kron(B.T, A, format='csr')
def s_H_1_norm(H):
    H_diag = H.diagonal()
    H_no_diag = H - np.diag(H_diag)
    N = len(H_diag)
    max_array = np.array(abs(H_no_diag).sum(axis=0))[0]
    
    n = 0
    for j in range(N):
        for i in range(N):
            n = max(n, max_array[i] + abs(H_diag[i] + np.conj(H_diag[j])) + max_array[j])
            
    return n
def get_spre_el(M, i, j, N):
    i_y, i_x = divmod(i,N)
    j_y, j_x = divmod(j,N)
    if j_y != i_y:
        return 0
    else:
        return M[i_x, j_x]

def get_spost_el(M, i, j, N):
    i_y, i_x = divmod(i,N)
    j_y, j_x = divmod(j,N)
    if j_x != i_x:
        return 0
    else:
        return M[i_y, j_y]
def L_1_norm(A, A_T, C_stack, C_stack_T):
    '''
    c_stack - vstack of c_list
    Norm of an operator IxA+A_TxI + sum C_TxC for C in C_list
    '''
    
    A_ptr = A.indptr
    A_ind = A.indices
    A_data = A.data
    
    A_T_ptr = A_T.indptr
    A_T_ind = A_T.indices
    A_T_data = A_T.data
    
    C_ptr = C_stack.indptr
    C_ind = C_stack.indices
    C_data = C_stack.data
    
    C_T_ptr = C_stack_T.indptr
    C_T_ind = C_stack_T.indices
    C_T_data = C_stack_T.data
    
    N = A.shape[0]
    
    arr_glob = np.zeros(N**2)
    arr_add = np.zeros(N**2)
    
    
    for i in range(N):
        for k in range(N):
            for j in range(A_ptr[k],A_ptr[k+1]):
                arr_add[N*i + A_ind[j]] += A_data[j]
                
            for j in range(A_T_ptr[i],A_T_ptr[i+1]):
                arr_T_add[N*i + A_T_ind[j]] += A_T_data[j]
            
            arr_glob += np.abs(arr_add)
            arr_add*=0
            
    
    return arr_glob
from py_mkl.py_mkl import py_mkl_sparse_z_mm, py_mkl_sparse_z_mv, py_mkl_sparse_d_mm, py_mkl_sparse_d_mv

def mkl_zspmm(A, B, alpha=1.0, beta = 0.0):
    """
    sparse matrix * dense matrix using MKL dcsrmm
    """
    (m,k) = A.shape
    (bk,n) = B.shape
    if k != bk:
        raise Exception('A and B dims are incompatible')

    # Allocate output, using same conventions as input    
    C = np.zeros((m,n),dtype=np.complex,order='C')
 
    py_mkl_sparse_z_mm(A.data, A.indptr, A.indices, B, C, alpha, beta)
    
    return C

def mkl_zspmv(A, B, alpha=1.0, beta = 0.0):
    """
    sparse matrix * dense matrix using MKL dcsrmm
    """
    (m,k) = A.shape
    (bk,n) = B.shape
    if n!=1:
        raise Exception('B has to be a vector')
    if k != bk:
        raise Exception('A and B dims are incompatible')
    n = B.shape[0]
    # Allocate output, using same conventions as input    
    C = np.zeros((m,1),dtype=np.complex,order='C')
 
    py_mkl_sparse_z_mv(A.data, A.indptr, A.indices, B, C, alpha, beta)
    
    return C


def mkl_dspmm(A, B, alpha=1.0, beta = 0.0):
    """
    sparse matrix * dense matrix using MKL dcsrmm
    """
    (m,k) = A.shape
    (bk,n) = B.shape
    if k != bk:
        raise Exception('A and B dims are incompatible')

    # Allocate output, using same conventions as input    
    C = np.zeros((m,n),dtype=np.float64,order='C')
 
    py_mkl_sparse_d_mm(A.data, A.indptr, A.indices, B, C, alpha, beta)
    
    return C

def mkl_dspmv(A, B, alpha=1.0, beta = 0.0):
    """
    sparse matrix * dense matrix using MKL dcsrmm
    """
    (m,k) = A.shape
    (bk,n) = B.shape
    if n!=1:
        raise Exception('B has to be a vector')
    if k != bk:
        raise Exception('A and B dims are incompatible')
    n = B.shape[0]
    # Allocate output, using same conventions as input    
    C = np.zeros((m,1),dtype=np.float64,order='C')
 
    py_mkl_sparse_d_mv(A.data, A.indptr, A.indices, B, C, alpha, beta)
    
    return C

def check_if_equel(*args):
    n = len(args)
    rand_ar_1 = np.random.rand(n)
    rand_ar_1 /= sum(rand_ar_1)

    rand_ar_2 = np.random.rand(n)
    rand_ar_2 /= sum(rand_ar_2)
    
    S = 0
    for i in range(n):
        S += rand_ar_1[i] * args[i] - rand_ar_2[i] * args[i]
    
    return np.linalg.norm(S)
