'''
Tools that are used to calculate covariance matrices of time aggregated income processes
'''
import numpy as np

def vech_indices(N):
    '''
    Returns the indices of the lower trianglular elements
    of an NxN matrix
    '''
    rows = [];
    columns = []
    for i in range(N):
        rows += range(i,N)
        columns += [i]*(N-i)
    return (np.array(rows), np.array(columns))

def vech(A):
    '''
    Returns the lower trianglular elements
    of an NxN matrix as a vector
    '''
    N = A.shape[0]
    indicies = vech_indices(N)
    return A[indicies]

def inv_vech(V):
    '''
    Inverse of vech. Returns a symetric matrix
    '''
    N = np.floor((len(V)*2)**0.5 ).astype(int)
    indicies = vech_indices(N)
    A = np.zeros((N,N))
    A[indicies] = V
    A[(indicies[1],indicies[0])] = V
    return A
