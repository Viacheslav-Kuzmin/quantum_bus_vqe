cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.stdlib cimport malloc, free
cdef extern from "complex.h":
    double complex conj(double complex x)
    
@cython.boundscheck(False)
@cython.cdivision(True)        
cdef void cy_unravel_inds( int ind, int[::1] dims, int * multy_inds ) nogil:
    
    cdef size_t j
    for j from dims.shape[0] > j >= 0:
        
        multy_inds[j] = ind % dims[j]
        ind = ind / dims[j]

cdef int[::1] cy_unrav_map(int[::1] dims):
    cdef int[::1] map_ar = np.ones(dims.shape[0], dtype=np.int32)
    
    cdef size_t j
    for j from dims.shape[0] > j >= 1:
        map_ar[j-1] = map_ar[j]*dims[j]
    
    return map_ar

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int cy_in(int val, int[::1] vec) nogil:
    # val in vec in pure cython
    cdef size_t ii
    for ii from 0 <= ii < vec.shape[0]:
        if val == vec[ii]:
            return 1
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cy_get_ellements_3(int *dims_I, int[::1] inds_I, int *dims_C, int[::1] inds_C, 
                      complex[:,:,::1] C_list, int[::1] unrav_map_A, size_t shape_C, size_t shape_I, 
                       complex[:,::1] coefs_list, int[:,::1] inds_A) nogil:
    
#     cdef size_t n_x, n_A, D, i, k, k1, k2, j, m, c1, c2, n_0, n_1 
#     cdef int d_dim
    cdef size_t n_x, n_A, D, i, k, k1, k2, j, m, c1, c2, n_0, n_1 
    cdef int d_dim
    
    n_x = 1
    n_A = 1
    for i from 0 <= i < inds_C.shape[0]:
        D = dims_C[i]
        d_dim = unrav_map_A[inds_C[i]]
        for k from 1 <= k < D:
            n_0 = (k-1)*n_A
            n_1 = k*n_A
            for j from 0 <= j < n_A:
                inds_A[0][n_1 + j] = inds_A[0][n_0 + j] + d_dim
        
        n_A = n_A*D

        for k1 from D > k1 >= 0:
            for c1 from n_x > c1 >= 0:
                for c2 from n_x > c2 >= 0:
                    for k2 from D > k2 >= 0:
                        coefs_list[k1*n_x+c1][k2*n_x + c2] = coefs_list[c1][c2] * C_list[i, k2, k1]
                        
        n_x = D*n_x
        
    n_A = 1
    n_1 = 1
    for i from 0 <= i < inds_I.shape[0]:
        D = dims_I[i]
        for k from 1 <= k < D:
            d_dim = k*unrav_map_A[inds_I[i]]
            for j from 0 <= j < n_A:
                for m from 0 <= m < shape_C:
                    inds_A[n_1][m] = inds_A[j][m] + d_dim
                n_1 += 1
        n_A = n_1

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_contract_parallel(complex[:,::1] A, int[::1] dims_A, int[::1] inds_I, complex[:,:,::1] C_list, int[::1] inds_C):
    
    '''
    Partial trace of indices inds_I + contraction of inds_C of A with operators in C_list
    '''
    
    cdef size_t shape_A, shape_B, shape_C, shape_I, i, j, y_B, x_B, i_y_A, i_x_A, n, ind_A, y_A, x_A
    
    # Shape of A
    shape_A = A.shape[0]
    cdef int *dims_C = <int *> malloc(inds_C.shape[0] * sizeof(int))
    cdef int *dims_I = <int *> malloc(inds_I.shape[0] * sizeof(int))
    cdef int *dims_CI = <int *> malloc((inds_C.shape[0] + inds_I.shape[0]) * sizeof(int))
    cdef int *inds_CI = <int *> malloc((inds_C.shape[0] + inds_I.shape[0]) * sizeof(int))

    for i from 0 <= i < inds_C.shape[0]:
        dims_C[i] = dims_A[inds_C[i]]
        dims_CI[i] = dims_A[inds_C[i]]
        inds_CI[i] = inds_C[i]
        
    for i from 0 <= i < inds_I.shape[0]:
        dims_I[i] = dims_A[inds_I[i]]
        dims_CI[inds_C.shape[0] + i] = dims_A[inds_I[i]]
        inds_CI[inds_C.shape[0] + i] = inds_I[i]
    
    shape_C = 1
    for i from 0 <= i < inds_C.shape[0]:        
        shape_C = shape_C* dims_C[i]
        
    shape_I = 1
    for i from 0 <= i < inds_I.shape[0]:        
        shape_I = shape_I * dims_I[i]

    # Properties of the state to remain
    n = 0
    cdef int * inds_B = <int *> malloc((dims_A.shape[0] - inds_I.shape[0]-inds_C.shape[0]) * sizeof(int))
    for i from 0 <= i < dims_A.shape[0]:
        if not cy_in(i, inds_C) and not cy_in(i, inds_I):
            inds_B[n] = i
            n += 1
    
    cdef np.ndarray[ np.int32_t, mode="c"]  dims_B = np.zeros(n, dtype = np.int32)
    shape_B = 1
    for i from 0 <= i < dims_B.shape[0]:
        dims_B[i] = dims_A[inds_B[i]]
        shape_B = shape_B * dims_B[i]
        
    cdef np.ndarray[ complex, ndim=2, mode="c"] data_B = np.zeros([shape_B, shape_B], dtype=np.complex128)
    
    # Map to convert a multyindex the to the index of the matrix
    cdef int[::1] unrav_map_A = cy_unrav_map(dims_A)
    
    cdef complex[:,::1] coefs_list = np.ones([shape_C,shape_C], dtype = complex)
    cdef int[:,::1] inds_A = np.zeros([shape_I,shape_C], dtype = np.int32)
    cy_get_ellements_3(dims_I, inds_I, dims_C, inds_C, C_list, unrav_map_A, 
                                            shape_C, shape_I, coefs_list, inds_A)
    
    # To get other elmenets of B we shift all indices to a required constant.
    cdef int[:,:,::1] inds_A_list = np.empty([shape_B,len(inds_A), len(inds_A[0])], dtype = np.int32)
#     cdef int[::1] B_multind = np.empty(inds_B.shape[0], dtype = np.int32)
    cdef int *B_multind = <int *> malloc(dims_B.shape[0] * sizeof(int))
    
    for ind_B from 0 <= ind_B < shape_B:
        cy_unravel_inds( ind_B, dims_B, B_multind )
        
        ind_A = 0
        for i from 0 <= i < dims_B.shape[0]:
            ind_A += B_multind[i]*unrav_map_A[inds_B[i]]
        
        for i from 0 <= i < inds_A.shape[0]:
            for j from 0 <= j < inds_A.shape[1]:
                inds_A_list[ind_B,i,j] = inds_A[i,j] + ind_A
    
    cdef int [:, :, ::1] inds_A_list_view = inds_A_list
    cdef int[::1] y_inds_A, x_A_list
    # Calculation of indices for upper triangle + diagonal
    for y_B in prange(shape_B, nogil=True):
#     for y_B from 0 <= y_B < shape_B:
        for i from 0 <= i < inds_A.shape[0]:
            
            for x_B from y_B <= x_B < shape_B:
                for i_y_A from 0 <= i_y_A < inds_A_list_view[y_B][i].shape[0]:
                    y_A = inds_A_list_view[y_B][i][i_y_A]
                    for i_x_A from 0 <= i_x_A < inds_A_list[x_B][i].shape[0]:
                        x_A = inds_A_list[x_B][i][i_x_A]
                        data_B[y_B, x_B] = data_B[y_B, x_B] + A[y_A, x_A] * coefs_list[i_y_A, i_x_A]
                        
#       Calculation of indices for lower triangle
    for y_B from 0 <= y_B < shape_B:
        for x_B from y_B <= x_B < shape_B:
            data_B[x_B, y_B] = conj(data_B[y_B, x_B])
              
    return data_B, dims_B




@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_contract(complex[:,::1] A, int[::1] dims_A, int[::1] inds_I, complex[:,:,::1] C_list, int[::1] inds_C):
    
    '''
    Partial trace of indices inds_I + contraction of inds_C of A with operators in C_list
    '''
    
    cdef size_t shape_A, shape_B, shape_C, shape_I, i, j, y_B, x_B, i_y_A, i_x_A, n, ind_A, y_A, x_A
    
    # Shape of A
    shape_A = A.shape[0]
    cdef int *dims_C = <int *> malloc(inds_C.shape[0] * sizeof(int))
    cdef int *dims_I = <int *> malloc(inds_I.shape[0] * sizeof(int))
    cdef int *dims_CI = <int *> malloc((inds_C.shape[0] + inds_I.shape[0]) * sizeof(int))
    cdef int *inds_CI = <int *> malloc((inds_C.shape[0] + inds_I.shape[0]) * sizeof(int))

    for i from 0 <= i < inds_C.shape[0]:
        dims_C[i] = dims_A[inds_C[i]]
        dims_CI[i] = dims_A[inds_C[i]]
        inds_CI[i] = inds_C[i]
        
    for i from 0 <= i < inds_I.shape[0]:
        dims_I[i] = dims_A[inds_I[i]]
        dims_CI[inds_C.shape[0] + i] = dims_A[inds_I[i]]
        inds_CI[inds_C.shape[0] + i] = inds_I[i]
    
    shape_C = 1
    for i from 0 <= i < inds_C.shape[0]:        
        shape_C = shape_C* dims_C[i]
        
    shape_I = 1
    for i from 0 <= i < inds_I.shape[0]:        
        shape_I = shape_I * dims_I[i]

    # Properties of the state to remain
    n = 0
    cdef int * inds_B = <int *> malloc((dims_A.shape[0] - inds_I.shape[0]-inds_C.shape[0]) * sizeof(int))
    for i from 0 <= i < dims_A.shape[0]:
        if not cy_in(i, inds_C) and not cy_in(i, inds_I):
            inds_B[n] = i
            n += 1
    
    cdef np.ndarray[ np.int32_t, mode="c"]  dims_B = np.zeros(n, dtype = np.int32)
    shape_B = 1
    for i from 0 <= i < dims_B.shape[0]:
        dims_B[i] = dims_A[inds_B[i]]
        shape_B = shape_B * dims_B[i]
        
    cdef np.ndarray[ complex, ndim=2, mode="c"] data_B = np.zeros([shape_B, shape_B], dtype=np.complex128)
    
    # Map to convert a multyindex the to the index of the matrix
    cdef int[::1] unrav_map_A = cy_unrav_map(dims_A)
    
    cdef complex[:,::1] coefs_list = np.ones([shape_C,shape_C], dtype = complex)
    cdef int[:,::1] inds_A = np.zeros([shape_I,shape_C], dtype = np.int32)
    cy_get_ellements_3(dims_I, inds_I, dims_C, inds_C, C_list, unrav_map_A, 
                                            shape_C, shape_I, coefs_list, inds_A)
    
    # To get other elmenets of B we shift all indices to a required constant.
    cdef int[:,:,::1] inds_A_list = np.empty([shape_B,len(inds_A), len(inds_A[0])], dtype = np.int32)
#     cdef int[::1] B_multind = np.empty(inds_B.shape[0], dtype = np.int32)
    cdef int *B_multind = <int *> malloc(dims_B.shape[0] * sizeof(int))
    
    for ind_B from 0 <= ind_B < shape_B:
        cy_unravel_inds( ind_B, dims_B, B_multind )
        
        ind_A = 0
        for i from 0 <= i < dims_B.shape[0]:
            ind_A += B_multind[i]*unrav_map_A[inds_B[i]]
        
        for i from 0 <= i < inds_A.shape[0]:
            for j from 0 <= j < inds_A.shape[1]:
                inds_A_list[ind_B,i,j] = inds_A[i,j] + ind_A
    
    cdef int[::1] y_inds_A, x_A_list
    # Calculation of indices for upper triangle + diagonal
    for y_B from 0 <= y_B < shape_B:
        for i from 0 <= i < inds_A.shape[0]:
            y_inds_A = inds_A_list[y_B][i]
            
            for x_B from y_B <= x_B < shape_B:
                for i_y_A from 0 <= i_y_A < y_inds_A.shape[0]:
                    y_A = y_inds_A[i_y_A]
                    x_A_list = inds_A_list[x_B][i]
                    for i_x_A from 0 <= i_x_A < x_A_list.shape[0]:
                        x_A = x_A_list[i_x_A]
                        data_B[y_B, x_B] = data_B[y_B, x_B] + A[y_A, x_A] * coefs_list[i_y_A, i_x_A]
                        
#       Calculation of indices for lower triangle
    for y_B from 0 <= y_B < shape_B:
        for x_B from y_B <= x_B < shape_B:
            data_B[x_B, y_B] = conj(data_B[y_B, x_B])
              
    return data_B, dims_B