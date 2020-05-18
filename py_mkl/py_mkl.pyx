import numpy as np
cimport numpy as np
cimport cython
    
cdef extern from "redefine_mkl_complex_16.h" nogil:
    pass

cdef extern from "mkl_spblas.h" nogil:
# cdef extern from "mkl_spblas.h" nogil:
    
    ctypedef enum sparse_status_t:
        pass
    ctypedef enum sparse_operation_t:
        SPARSE_OPERATION_NON_TRANSPOSE = 10
    
    ctypedef enum sparse_layout_t:
        SPARSE_LAYOUT_ROW_MAJOR = 101
    
    ctypedef enum sparse_index_base_t:
        SPARSE_INDEX_BASE_ZERO = 0
    
    ctypedef enum sparse_matrix_type_t:
        SPARSE_MATRIX_TYPE_GENERAL            = 20
        SPARSE_MATRIX_TYPE_SYMMETRIC          = 21
        SPARSE_MATRIX_TYPE_HERMITIAN          = 22
        SPARSE_MATRIX_TYPE_TRIANGULAR         = 23
        SPARSE_MATRIX_TYPE_DIAGONAL           = 24
        SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR   = 25
        SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL     = 26
    
    ctypedef enum sparse_fill_mode_t:
        SPARSE_FILL_MODE_LOWER  = 40
        SPARSE_FILL_MODE_UPPER  = 41
        SPARSE_FILL_MODE_FULL  = 42 
    
    ctypedef enum sparse_diag_type_t:
        SPARSE_DIAG_NON_UNIT    = 50
        SPARSE_DIAG_UNIT        = 51
    
    ctypedef enum sparse_matrix_type_t:
        pass
    
    ctypedef enum sparse_fill_mode_t:
        pass
    
    ctypedef enum sparse_diag_type_t:
        pass
    
    cdef struct matrix_descr:
        sparse_matrix_type_t  type
        sparse_fill_mode_t    mode
        sparse_diag_type_t    diag
        
    cdef struct sparse_matrix
    ctypedef sparse_matrix *sparse_matrix_t
    ctypedef struct sparse_index_base_t
    
    sparse_status_t mkl_sparse_destroy( sparse_matrix_t  A )
        
    sparse_status_t mkl_sparse_z_create_csr( sparse_matrix_t *A, sparse_index_base_t indexing, int rows, int cols, int *rows_start, int *rows_end, int *col_indx, complex *values )
    
    sparse_status_t mkl_sparse_d_create_csr( sparse_matrix_t *A, sparse_index_base_t indexing, int rows, int cols, int *rows_start, int *rows_end, int *col_indx, double *values )
    
    sparse_status_t mkl_sparse_z_mm (sparse_operation_t op, complex alpha, sparse_matrix_t A, matrix_descr descr, sparse_layout_t layout, complex *x, int columns, int ldx, complex beta, complex *y, int ldy)
    
    sparse_status_t mkl_sparse_z_mv (sparse_operation_t operation, complex alpha, sparse_matrix_t A, matrix_descr descr, complex *x, complex beta, complex *y)
    
    
    sparse_status_t mkl_sparse_d_mm (sparse_operation_t op, double alpha, sparse_matrix_t A, matrix_descr descr, sparse_layout_t layout, double *x, int columns, int ldx, double beta, double *y, int ldy)
    
    sparse_status_t mkl_sparse_d_mv (sparse_operation_t operation, double alpha, sparse_matrix_t A, matrix_descr descr, double *x, double beta, double *y)

    
cdef extern from "mkl.h" nogil:
# cdef extern from "mkl.h" nogil:
    ctypedef enum CBLAS_LAYOUT:
        CblasRowMajor=101
        CblasColMajor=102
    ctypedef enum CBLAS_TRANSPOSE:
        CblasNoTrans=111,
        CblasTrans=112,
        CblasConjTrans=113
    
    void mkl_zomatadd (char ordering, char transa, char transb, size_t m, size_t n, complex alpha, complex * A, size_t lda, complex beta, complex * B, size_t ldb, complex * C, size_t ldc)
    
    void cblas_zgemm (CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, int m, int n, int k, complex *alpha, complex *a, int lda, complex *b, int ldb, complex *beta, complex *c, int ldc)
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void py_cblas_zgemm( complex[::,::1] A, complex[::,::1] B, complex[::,::1] C):
    
    cdef CBLAS_LAYOUT Layout = CblasRowMajor
    cdef CBLAS_TRANSPOSE trans = CblasNoTrans
    cdef int m = A.shape[0]
    cdef int n = B.shape[1]
    cdef int k = B.shape[0]
    cdef complex alpha = 1
    cdef complex beta = 0
    
    cblas_zgemm(Layout, trans, trans, m, n, k, &alpha, &A[0,0], k, &B[0,0], n, &beta, &C[0,0], n )
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void py_cblas_zgemm_vec( complex[::,::1] A, complex[::,::,::1] B, complex[::,::,::1] C):
    
    cdef CBLAS_LAYOUT Layout = CblasRowMajor
    cdef CBLAS_TRANSPOSE trans = CblasNoTrans
    cdef int m = A.shape[0]
    cdef int n = B.shape[2]
    cdef int k = B.shape[1]
    cdef int N_b = B.shape[0]
    cdef complex alpha = 1
    cdef complex beta = 0
    
    cdef size_t v
    
    for v from 0 <= v < N_b :
        cblas_zgemm(Layout, trans, trans, m, n, k, &alpha, &A[0,0], k, &B[v,0,0], n, &beta, &C[v,0,0], n )
    
    
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void update_matrix_z( sparse_matrix_t *A, int Ndim, int *ptr_0, int *inds_0, complex *data_0 ):
    mkl_sparse_z_create_csr( A, SPARSE_INDEX_BASE_ZERO, Ndim, Ndim, ptr_0, ptr_0+1, inds_0, data_0)
    

@cython.boundscheck(True)
@cython.wraparound(True)
cdef void update_matrix_d( sparse_matrix_t *A, int Ndim, int *ptr_0, int *inds_0, double *data_0 ):
    mkl_sparse_d_create_csr( A, SPARSE_INDEX_BASE_ZERO, Ndim, Ndim, ptr_0, ptr_0+1, inds_0, data_0)
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void py_mkl_sparse_z_mm( complex[::1] A_data, int[::1] A_ptr, int[::1] A_inds, complex[::,::1] B, complex[::,::1] C, complex alpha, complex beta):
    
    cdef sparse_matrix_t csrA
    cdef matrix_descr descr = matrix_descr(SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT)

    cdef int ldy, columns
    ldy = C.shape[0]
    columns = C.shape[1]
    
    update_matrix_z( &csrA, ldy, &A_ptr[0], &A_inds[0], &A_data[0])
    
    mkl_sparse_z_mm( SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descr, SPARSE_LAYOUT_ROW_MAJOR, &B[0,0], columns, ldy, beta, &C[0,0], ldy )
    mkl_sparse_destroy ( csrA )
  
 
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void py_mkl_sparse_z_mv( complex[::1] A_data, int[::1] A_ptr, int[::1] A_inds, complex[::,::1] B, complex[::,::1] C, complex alpha, complex beta):
    
    cdef sparse_matrix_t csrA
    cdef matrix_descr descr = matrix_descr(SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT)

    cdef int ldy = C.shape[0]
    
    update_matrix_z( &csrA, ldy, &A_ptr[0], &A_inds[0], &A_data[0])
    
    mkl_sparse_z_mv( SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descr, &B[0,0], beta, &C[0,0])
    mkl_sparse_destroy ( csrA )
  


    
@cython.boundscheck(True)
@cython.wraparound(True)
cpdef void py_mkl_sparse_d_mm( double[::1] A_data, int[::1] A_ptr, int[::1] A_inds, double[::,::1] B, double[::,::1] C, double alpha, double beta):
    
    cdef sparse_matrix_t csrA
    cdef matrix_descr descr = matrix_descr(SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT)

    cdef int ldy, columns
    ldy = C.shape[0]
    columns = C.shape[1]
    
    update_matrix_d( &csrA, ldy, &A_ptr[0], &A_inds[0], &A_data[0])
    
    mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descr, SPARSE_LAYOUT_ROW_MAJOR, &B[0,0], columns, ldy, beta, &C[0,0], ldy )
    mkl_sparse_destroy ( csrA )
  
 
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void py_mkl_sparse_d_mv( double[::1] A_data, int[::1] A_ptr, int[::1] A_inds, double[::,::1] B, double[::,::1] C, double alpha, double beta):
    
    cdef sparse_matrix_t csrA
    cdef matrix_descr descr = matrix_descr(SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT)

    cdef int ldy = C.shape[0]
    
    update_matrix_d( &csrA, ldy, &A_ptr[0], &A_inds[0], &A_data[0])
    
    mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descr, &B[0,0], beta, &C[0,0])
    mkl_sparse_destroy ( csrA )
   