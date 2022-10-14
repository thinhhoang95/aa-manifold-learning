import numpy as np 
from scipy import sparse as sparse
from scipy.sparse import linalg as slinalg

from scipy.linalg import svd as lsvd

def SVTs(X, tau, svd_k=6):
    """Perform Singular Value Thresholding for a Sparse Matrix

    Args:
        X (np.array): the SPARSE matrix to be thresholded
        tau (number): the parameter associated with the nuclear norm

    Returns:
        np.ndarray: SVTed matrix
    """
    # Convert to sparse matrix Xs, s stands for sparse
    Xs = sparse.csc_matrix(X)
    # SVD decomposition
    u, s, vh = slinalg.svds(Xs, k=svd_k)
    # Shrink the singular values by tau:
    s_new = sparse.diags(np.maximum(0,s-tau))
    return u @ s_new @ vh

def SVT(X, tau):
    """Perform Singular Value Thresholding for a Normal Matrix

    Args:
        X (np.ndarray): The matrix to be SVT'd
        tau (float): The value associated with the nuclear norm

    Returns:
        np.ndarray: SVT'd matrix
    """
    # SVD decomposition
    u, s, vh = lsvd(X)
    # Shrink the singular values by tau:
    s_new = np.zeros((u.shape[0], vh.shape[0]))
    np.fill_diagonal(s_new, np.maximum(0,s-tau))
    return u @ s_new @ vh

def evalSparseObjFn(X: np.array, Y: np.array, tau: float):
    """Calculate the expression 1/2||X-Y||_F^2 + tau ||X||*
    
    Args:
        X (np.array): the thresholded matrix
        Y (np.array): the original matrix

    Returns:
        number: the objective function 1/2||X-Y||_F^2 + tau ||X||*
    """

    fr_term = 0.5 * np.linalg.norm(X-Y, ord="fro") # Frobenius norm
    u, s, vh = np.linalg.svd(X)
    nn_term = np.sum(s)

    return fr_term+nn_term

def get_spectrum(X, max_k = 3):
    u, s, vh = lsvd(X)
    return(s[:max_k])