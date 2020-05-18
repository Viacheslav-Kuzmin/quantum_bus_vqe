
# coding: utf-8

# In[4]:


from qutip import *
import numpy as np
import scipy as sp


# In[7]:


from py_mkl.py_mkl import py_mkl_sparse_z_mm, py_mkl_sparse_z_mv, py_mkl_sparse_d_mm, py_mkl_sparse_d_mv, py_cblas_zgemm, py_cblas_zgemm_vec


# # Old methods

# In[5]:


from ctypes import POINTER,c_int,c_char,c_double, byref
from numpy import ctypeslib
import qutip.settings as qset

dcsrmm = qset.mkl_lib.mkl_dcsrmm

def mkl_dspmm_old(A, B, alpha=1.0, beta = 0.0):
    """
    sparse matrix * dense matrix using MKL dcsrmm
    """
    (m,k) = A.shape
    (bk,n) = B.shape
    if k != bk:
        raise Exception('A and B dims are incompatible')

    # Allocate output, using same conventions as input    
    C = np.zeros((m,n),dtype=np.float64,order='C')

    np_B = B.ctypes.data_as(ctypeslib.ndpointer(np.float64, ndim=2, flags='C'))
    np_C = C.ctypes.data_as(ctypeslib.ndpointer(np.float64, ndim=2, flags='C'))

    # Pointers to data of the matrix
    data    = A.data.ctypes.data_as(ctypeslib.ndpointer(np.float64, ndim=1, flags='C'))
    pointerB = A.indptr[:-1]
    pointerE = A.indptr[1:]
    np_pointerB = pointerB.ctypes.data_as(POINTER(c_int))
    np_pointerE = pointerE.ctypes.data_as(POINTER(c_int))
    indices = A.indices.ctypes.data_as(POINTER(c_int))

    matdescra = np.chararray(6)
    matdescra[0] = 'G'#G-general, S-symmetric, H-hermitian
    matdescra[1] = 'L'
    matdescra[2] = 'N'
    matdescra[3] = 'C'
    np_matdescra = matdescra.ctypes.data_as(POINTER(c_char))

    # now call MKL
    dcsrmm(byref(c_char(bytes(b'N'))), 
          byref(c_int(m)),
          byref(c_int(n)),
          byref(c_int(k)),
          byref(c_double(alpha)),
          np_matdescra,
          data,
          indices,
          np_pointerB,
          np_pointerE,
          np_B,
          byref(c_int(n)),
          byref(c_double(beta)),
          np_C,
          byref(c_int(n))) 

    return C


zcsrmm = qset.mkl_lib.mkl_zcsrmm

def mkl_zspmm_old(A, B, alpha=1.0, beta = 0.0):
    """
    sparse matrix * dense matrix using MKL dcsrmm
    """
    (m,k) = A.shape
    (bk,n) = B.shape
    if k != bk:
        raise Exception('A and B dims are incompatible')

    # Allocate output, using same conventions as input    
    C = np.zeros((m,n),dtype=np.complex,order='C')

    np_B = B.ctypes.data_as(ctypeslib.ndpointer(np.complex, ndim=2, flags='C'))
    np_C = C.ctypes.data_as(ctypeslib.ndpointer(np.complex, ndim=2, flags='C'))

    # Pointers to data of the matrix
    data    = A.data.ctypes.data_as(ctypeslib.ndpointer(np.complex, ndim=1, flags='C'))
    pointerB = A.indptr[:-1]
    pointerE = A.indptr[1:]
    np_pointerB = pointerB.ctypes.data_as(POINTER(c_int))
    np_pointerE = pointerE.ctypes.data_as(POINTER(c_int))
    indices = A.indices.ctypes.data_as(POINTER(c_int))

    matdescra = np.chararray(6)
    matdescra[0] = 'G'#G-general, S-symmetric, H-hermitian
    matdescra[1] = 'L'
    matdescra[2] = 'N'
    matdescra[3] = 'C'
    np_matdescra = matdescra.ctypes.data_as(POINTER(c_char))

    # now call MKL
    zcsrmm(byref(c_char(bytes(b'N'))), 
          byref(c_int(m)),
          byref(c_int(n)),
          byref(c_int(k)),
          byref(c_double(alpha)),
          np_matdescra,
          data,
          indices,
          np_pointerB,
          np_pointerE,
          np_B,
          byref(c_int(n)),
          byref(c_double(beta)),
          np_C,
          byref(c_int(n))) 

    return C

zcsrgemv = qset.mkl_lib.mkl_cspblas_zcsrgemv

def mkl_zspmv_old(A, x):
    """
    sparse csr_spmv using MKL
    """
    (m,n) = A.shape

    # Pointers to data of the matrix
    data = A.data.ctypes.data_as(ctypeslib.ndpointer(np.complex128, ndim=1, flags='C'))
    indptr = A.indptr.ctypes.data_as(POINTER(c_int))
    indices = A.indices.ctypes.data_as(POINTER(c_int))

    # Allocate output, using same conventions as input
    if x.ndim is 1:
        y = np.empty(m,dtype=np.complex, order='C')
    elif x.ndim==2 and x.shape[1]==1:
        y = np.empty((m,1),dtype=np.complex, order='C')
    else:
        raise Exception('Input vector must be 1D row or 2D column vector')

    np_x = x.ctypes.data_as(ctypeslib.ndpointer(np.complex128, ndim=1, flags='C'))
    np_y = y.ctypes.data_as(ctypeslib.ndpointer(np.complex128, ndim=1, flags='C'))

    # now call MKL. This returns the answer in np_y, which points to y
    zcsrgemv(byref(c_char(bytes(b'N'))), byref(c_int(m)), data ,indptr, indices, np_x, np_y ) 
    return y

dcsrgemv = qset.mkl_lib.mkl_cspblas_dcsrgemv
def mkl_dspmv_old(A, x, is_trnsa = False):
    """
    sparse csr_spmv using MKL
    """
    (m,n) = A.shape

    # Pointers to data of the matrix
    data = A.data.ctypes.data_as(ctypeslib.ndpointer(np.float64, ndim=1, flags='C'))
    indptr = A.indptr.ctypes.data_as(POINTER(c_int))
    indices = A.indices.ctypes.data_as(POINTER(c_int))

    # Allocate output, using same conventions as input
    if x.ndim is 1:
        y = np.empty(m,dtype=np.float64, order='C')
    elif x.ndim==2 and x.shape[1]==1:
        y = np.empty((m,1),dtype=np.float64, order='C')
    else:
        raise Exception('Input vector must be 1D row or 2D column vector')

    np_x = x.ctypes.data_as(ctypeslib.ndpointer(np.float64, ndim=1, flags='C'))
    np_y = y.ctypes.data_as(ctypeslib.ndpointer(np.float64, ndim=1, flags='C'))

    
    if is_trnsa:
        transa = c_char(bytes(b'T'))
    else:
        transa = c_char(bytes(b'N'))
        
    # now call MKL. This returns the answer in np_y, which points to y
    dcsrgemv(byref(transa), byref(c_int(m)), data ,indptr, indices, np_x, np_y ) 
    return y


# # New methods

# In[8]:


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

def cblas_zgemm(A, B, alpha=1.0, beta = 0.0):
    """
    sparse matrix * dense matrix using MKL dcsrmm
    """
    (m,k) = A.shape
    (bk,n) = B.shape
    if k != bk:
        raise Exception('A and B dims are incompatible')
    n = B.shape[0]
    # Allocate output, using same conventions as input    
    C = np.zeros((m,n),dtype=complex,order='C')
 
    py_cblas_zgemm(A, B, C)
    
    return C

def cblas_zgemm_vec(A, B_vec, alpha=1.0, beta = 0.0):
    """
    sparse matrix * dense matrix using MKL dcsrmm
    """
#     (m,k) = A.shape
#     (bk,n) = B.shape
#     if k != bk:
#         raise Exception('A and B dims are incompatible')
#     n = B.shape[0]
    # Allocate output, using same conventions as input    
    C_vec = np.zeros(B_vec.shape,dtype=complex,order='C')
 
    py_cblas_zgemm_vec(A, B_vec, C_vec)
    
    return C_vec

def check_if_equal(*args):
    n = len(args)
    rand_ar_1 = np.random.rand(n)
    rand_ar_1 /= sum(rand_ar_1)

    rand_ar_2 = np.random.rand(n)
    rand_ar_2 /= sum(rand_ar_2)
    
    S = 0
    for i in range(n):
        S += rand_ar_1[i] * args[i] - rand_ar_2[i] * args[i]
    
    return np.linalg.norm(S)


# # Test dense

# In[4]:


N = 2000
A = np.random.rand(N,N)*(1+0.j)
B = np.random.rand(N,N)*(1+0.j)


# In[5]:


C = np.dot(A,B)
C_mkl = cblas_zgemm(A, B)
np.linalg.norm(C_mkl-C)


# In[6]:


get_ipython().run_line_magic('timeit', 'cblas_zgemm(A, B)')
get_ipython().run_line_magic('timeit', 'np.dot(A,B)')


# ## Multiple product

# In[24]:


N = 100
N_b = 100
A = np.random.rand(N,N)*(1+0.j)
B_vec = np.random.rand(N_b,N,N)*(1+0.j)


# In[25]:


C_list = [np.dot(A,B) for B in B_vec]
C_mkl = cblas_zgemm_vec(A, B_vec)
sum(np.linalg.norm(C_mkl[i]-C_list[i]) for i in range(N_b))


# In[26]:


get_ipython().run_line_magic('timeit', '[np.dot(A,B) for B in B_vec]')
get_ipython().run_line_magic('timeit', 'cblas_zgemm_vec(A, B_vec)')


# # Test sparse

# In[4]:


N = 1000
mat_dense_d = np.random.rand(N,N)
vec_dense_d = np.random.rand(N,1)
mat_sparse_d = sp.sparse.random(N,N,format='csr')

mat_dense_z = mat_dense_d * (1.+0.j)
vec_dense_z = vec_dense_d * (1.+0.j)
mat_sparse_z = mat_sparse_d * (1.+0.j)


# In[7]:


check_if_equal(
    mkl_zspmm(mat_sparse_z, vec_dense_z),
    mkl_zspmv(mat_sparse_z, vec_dense_z),
#     mkl_dspmm(mat_sparse_d, vec_dense_d),
    mkl_dspmv(mat_sparse_d, vec_dense_d)
)


# In[ ]:


check_if_equal(
    mkl_zspmm(mat_sparse_z, mat_dense_z),
    mkl_dspmm(mat_sparse_d, mat_dense_d)
)


# In[54]:


get_ipython().run_line_magic('timeit', 'mkl_zspmm(mat_sparse_z, vec_dense_z)')
get_ipython().run_line_magic('timeit', 'mkl_zspmv(mat_sparse_z, vec_dense_z)')
# %timeit mkl_dspmm(mat_sparse_d, vec_dense_d)
get_ipython().run_line_magic('timeit', 'mkl_dspmv(mat_sparse_d, vec_dense_d)')


# In[8]:


get_ipython().run_line_magic('timeit', 'mkl_zspmm(mat_sparse_z, vec_dense_z)')
get_ipython().run_line_magic('timeit', 'mkl_zspmv(mat_sparse_z, vec_dense_z)')
# %timeit mkl_dspmm(mat_sparse_d, vec_dense_d)
get_ipython().run_line_magic('timeit', 'mkl_dspmv(mat_sparse_d, vec_dense_d)')


# In[55]:


get_ipython().run_line_magic('timeit', 'mkl_zspmm(mat_sparse_z, mat_dense_z)')
get_ipython().run_line_magic('timeit', 'mkl_dspmm(mat_sparse_d, mat_dense_d)')


# In[9]:


get_ipython().run_line_magic('timeit', 'mkl_zspmm(mat_sparse_z, mat_dense_z)')
get_ipython().run_line_magic('timeit', 'mkl_dspmm(mat_sparse_d, mat_dense_d)')


# In[ ]:


mkl_dspmm(mat_sparse_d, mat_dense_d)


# In[ ]:




















