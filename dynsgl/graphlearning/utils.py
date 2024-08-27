import numpy as  np

from scipy import sparse

def rowsum_mat(n):
    """ Returns a matrix which can be used to find row-sum of a symmetric matrix 
    with zero diagonal from its vectorized upper triangular part. 
    
    For nxn symmetric zero-diagonal matrix A, let a be its M=n(n-1)/2 dimensional 
    vector of upper triangular part. Row-sum matrix S is nxM dimensional matrix 
    such that:
    .. math:: Sa = A1,
    where 1 is n dimensional all-one vector.
    
    Parameters
    ----------
    n : int
        Dimension of the matrix.
    Returns
    -------
    S : sparse matrix
        Matrix to be used in row-sum calculation
    """

    i, j = np.triu_indices(n, k=1)
    M = len(i)
    rows = np.concatenate((i, j))
    cols = np.concatenate((np.arange(M), np.arange(M)))

    return sparse.csr_matrix((np.ones((2*M, )), (rows, cols)), shape=(n, M))

def temporal_diff_mat(n_times):
    rows = np.concatenate((np.arange(n_times-1), np.arange(n_times-1)))
    cols = np.concatenate((np.arange(n_times-1), np.arange(n_times-1)+1))
    data = np.concatenate((-1*np.ones(n_times-1), np.ones(n_times-1)))

    return sparse.csr_matrix((data, (rows, cols)), shape=(n_times-1, n_times))