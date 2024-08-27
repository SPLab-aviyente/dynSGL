import time

import numpy as np

from numba import njit

from sklearn.metrics import jaccard_score

from . import utils

@njit
def _update_auxiliaries(y_pos, y_neg):
    # Projection onto complementarity set
    n_pairs = len(y_pos)
    v = np.zeros((n_pairs, 1))
    w = np.zeros((n_pairs, 1))
    for i in range(n_pairs):
        if y_pos[i] < 0 and y_neg[i] < 0:
            if y_pos[i] < y_neg[i]:
                v[i] = y_pos[i]
            else:
                w[i] = y_neg[i]
        elif y_pos[i] < 0 and y_neg[i] >= 0:
            v[i] = y_pos[i]
        elif y_pos[i] >= 0 and y_neg[i] < 0:
            w[i] = y_neg[i]

    return v, w

def _project_to_hyperplane(v, n):
    return v - (n + np.sum(v))/(len(v))

def _update_laplacian(data_vec, l_prev, l_next, v, y, S, alpha, beta, beta_next, 
                      rho, n):
    y = 4*beta*l_prev + 4*beta_next*l_next - data_vec + rho*v - y

    a = 4*alpha + 4*beta + 4*beta_next + rho
    b = 2*alpha
    c1 = 1/a
    c2 = b/(a*(a+n*b-2*b))
    c3 = (4*b**2)/(a*(a+(n-2)*b)*(a+2*(n-1)*b))

    y = c1*y - c2*(S.T@(S@y)) + c3*np.sum(y)

    return _project_to_hyperplane(y, n)

def _objective(data_vecs, l, alpha, beta, S):
    n_times = len(data_vecs)

    result = 0
    sign = {"+": 1, "-": -1} # for calculating smoothness
    for s in ["+", "-"]:
        if alpha[s] is not None:
            for t in range(n_times):
                result += sign[s]*(data_vecs[t]).T@l[s][t] # smoothness
                result += alpha[s][t]*np.linalg.norm(S@l[s][t])**2 # degree term
                result += 2*alpha[s][t]*np.linalg.norm(l[s][t])**2 # sparsity term
            
                if t > 0:
                    result += 2*beta[s][t-1]*np.linalg.norm(l[s][t] - l[s][t-1])**2 # temporal smoothness

    return result.item()

def _run(data_vecs, alpha, beta, S, rho=10, max_iter=100):
    n_times = len(data_vecs)
    n_pairs = int(len(data_vecs[0])) # number of node pairs
    n_nodes = int((1 + np.sqrt(8*n_pairs+1))//2) # number of nodes

    # Initialization
    v = {"+": [None]*n_times, "-": [None]*n_times}
    l = {"+": [], "-": []}
    y = {"+": [], "-": []}
    for s in ["+", "-"]:
        for t in range(n_times):
            l[s].append(np.zeros((n_pairs, 1)))
            y[s].append(np.zeros((n_pairs, 1)))

    # Iterations
    objective_vals = []
    for iter in range(max_iter):

        # Update auxiliary variables
        for t in range(n_times):
            y_pos = l["+"][t] + y["+"][t]/rho
            y_neg = l["-"][t] + y["-"][t]/rho
            v["+"][t], v["-"][t] = _update_auxiliaries(y_pos, y_neg)

        # Update laplacians
        sign = {"+": 1, "-": -1} 
        for s in ["+", "-"]:
            if alpha[s] is not None:
                l[s][0] = _update_laplacian(sign[s]*data_vecs[0], 0, l[s][1], v[s][0], y[s][0],
                                            S, alpha[s][0], 0, beta[s][0], rho, n_nodes)
                
                for t in range(1, n_times-1):
                    l[s][t] = _update_laplacian(sign[s]*data_vecs[t], l[s][t-1], l[s][t+1], 
                                                v[s][t], y[s][t], S, alpha[s][t], 
                                                beta[s][t-1], beta[s][t], rho, n_nodes)

                t = n_times-1
                l[s][t] = _update_laplacian(sign[s]*data_vecs[t], l[s][t-1], 0, v[s][t], y[s][t],
                                            S, alpha[s][t], beta[s][t-1], 0, rho, n_nodes)

        # Update multipliers
        for s in ["+", "-"]:
            if alpha[s] is not None:
                for t in range(n_times):
                    y[s][t] += rho*(l[s][t] - v[s][t])

        objective_vals.append(_objective(data_vecs, v, alpha, beta, S))

        if iter > 10 and abs(objective_vals[-1] - objective_vals[-2]) < 1e-4:
            break
        
    
    # Remove small edges and convert to the adjacency matrix
    for s in ["+", "-"]:
        for t in range(n_times):
            v[s][t][v[s][t]>-1e-4] = 0
            v[s][t] = np.abs(v[s][t])
  
    return v

def _similarity(w, w_prev):
    # return np.sum((w>0) & (w_prev>0))/np.sum(w>0)
    return np.corrcoef(np.squeeze(w_prev), np.squeeze(w))[0,1]

def _density(w):
    return np.count_nonzero(w)/len(w)

def learn_a_dynamic_signed_graph(X, alpha_pos, alpha_neg, beta_pos, beta_neg, 
                                  density = None, similarity = None, 
                                  param_acc = 0.025, n_iters = 50, **kwargs):
    r"""Learn a dynamic signed graph from temporal graph signals.

    Assume we are given a set of data matrices :math:`\{\mathbf{X}^t\}_{t=1}^T`,
    where each data matrix includes graph signals defined on unknown dynamic 
    signed graph :math:`\{G^t\}_{t=1}^T`. This function learns
    :math:`G^t`'s by minimizing smoothness of :math:`X^t`'s with respect to
    positive Laplacian matrices, while maximizing smoothness with respect to 
    negative Laplacian matrices. During learning, it is assumed :math:`G^t`
    is similar to :math:`G^{t-1}`. 

    The function also has an experimental hyperparameter search procedure which
    tries to optimize hyperparameters of the learning algorithm to obtain graphs
    with desired properties. By default this procedure isn't applied if its
    parameters are not set.

    .. warning::
        The hyperparameter search procedure is experimental and hasn't been 
        tested extensively. It might not return expected output, so use with care.

    Parameters
    ----------
    X : list of np.array
        Data matrices.
    alpha_pos : float
        The hyperparameter controling the density of the learned :math:`G^{t, +}`'s.
        Its larger values learn denser graphs. If 0, graphs with only negative 
        edges are learned.
    alpha_neg : float
        The hyperparameter controling the density of the learned :math:`G^{t, -}`'s.
        Its larger values learn denser graphs. If 0, graphs with only positive 
        edges are learned.
    beta_pos : float
        The hyperparameter controling how similar the learned :math:`G^{t, +}` 
        and :math:`G^{t-1, +}` are. Its larger values impose more similarity.
    beta_pos : float
        The hyperparameter controling how similar the learned :math:`G^{t, -}` 
        and :math:`G^{t-1, -}` are. Its larger values impose more similarity.
    density : float, optional
        The hyperparameter search procedure optimizes `alpha_pos` and `alpha_neg` 
        to make density of learned :math:`G^{t, +}` and :math:`G^{t, -}` to be 
        close to this value. If None, hyperparameters aren't optimized. 
        By default None.
    similarity : float, optional
        The hyperparameter search procedure optimizes `beta_pos` (`beta_neg`) 
        to make similarity of learned :math:`G^{t,+}` and :math:`G^{t-1,+}` (
        :math:`G^{t,-}` and :math:`G^{t-1,-}`) to be close to this value.
        Similarity is calculated using correlation between adjacencies. If None,
        hyperparameters aren't optimized. By default None.
    param_acc : float, optional
        Hyperparameter search procedure assumed to be converged when the
        difference between desired and learned graph properties is smaller than
        this value., by default 0.025
    n_iters : int, optional
        Maximum number of iterations for hyperparameter search, by default 50

    Returns
    -------
    w_hat : dict of list of np.array
        w_hat["+"] includes upper triangular part of adjacency matrices of 
        learned :math:`G^{t, +}` and w_hat["-"] includes those of :math:`G^{t, -}`.
    run_time : float
        Time passed to learn the dynamic signed graph for given values of hyperparameters.

    """

    # Input checks
    if not isinstance(X, list):
        raise Exception("Multiple sets of graph signals must be provided when "
                        "learning a dynamic signed graph.")
    
    # Variable initialization
    n_times = len(X)
    n_nodes = X[0].shape[0]
    S = utils.rowsum_mat(n_nodes)
    
    input_alpha = {}
    input_beta = {}
    alpha = {}
    if alpha_pos == 0:
        beta_pos = 0
        input_alpha["+"] = None
        alpha["+"] = None
    else:
        input_alpha["+"] = alpha_pos
        alpha["+"] = alpha_pos*np.ones(n_times)

    if alpha_neg == 0:
        beta_neg = 0
        input_alpha["-"] = None
        alpha["-"] = None
    else:
        input_alpha["-"] = alpha_neg
        alpha["-"] = alpha_neg*np.ones(n_times)

    input_beta = {"+": beta_pos, "-": beta_neg}
    beta = {"+": beta_pos*np.ones(n_times-1), 
            "-": beta_neg*np.ones(n_times-1)}

    # Data preparation: Get 2k - S^T@d for each time point
    data_vecs = []
    for t in range(n_times):
        K = X[t]@X[t].T
        k = K[np.triu_indices_from(K, k=1)]
        d = K[np.diag_indices_from(K)]
        data_vecs.append(2*k - S.T@d)
        if np.ndim(data_vecs[-1]) == 1:
            data_vecs[-1] = data_vecs[-1][:, None]

        data_vecs[-1] /= np.max(np.abs(data_vecs[-1]))

    densities = {"+": [[] for i in range(n_times)], "-": [[] for i in range(n_times)]}
    similarities = {"+": [[] for i in range(n_times-1)], "-": [[] for i in range(n_times-1)]}

    for iter_indx in range(n_iters):

        st = time.time()
        w = _run(data_vecs, alpha, beta, S, **kwargs)
        run_time = time.time() - st

        are_params_updated = False

        if density is not None:
            iter_densities = {"+": [], "-": []}
            for s in ["+", "-"]:
                if input_alpha[s] is not None:
                    for t in range(n_times):
                        density_hat = _density(w[s][t])

                        diff = density_hat - density
                        if np.abs(diff) > param_acc:
                            alpha[s][t] *= density/density_hat
                            are_params_updated = True

                        densities[s][t].append(density_hat)
                        iter_densities[s].append(density_hat)
        
        if similarity is not None:
            iter_similarities = {"+": [], "-": []}
            for s in ["+", "-"]:
                if input_beta[s] > 0:
                    for t in range(1, n_times):
                        similarity_hat = _similarity(w[s][t], w[s][t-1])

                        diff = similarity_hat - similarity
                        if np.abs(diff) > param_acc:
                            beta[s][t-1] *= similarity/similarity_hat
                            are_params_updated = True

                        similarities[s][t-1].append(similarity_hat)
                        iter_similarities[s].append(similarity_hat)

        if not are_params_updated:
            break

        if iter_indx > 6:
            converged = True 
            if density is not None:
                for s in ["+", "-"]:
                    if input_alpha[s] is not None:
                        for t in range(n_times):
                            converged = (
                                converged and 
                                (np.mean(densities[s][t][-5:]) - iter_densities[s][t]) < 1e-4
                            )
            if similarity is not None:
                for s in ["+", "-"]:
                    if input_beta[s] > 0:
                        for t in range(n_times-1):
                            converged = (
                                converged and
                                (np.mean(similarities[s][t][-5:] - iter_similarities[s][t])) < 1e-4
                            )

            if converged: 
                break

    return w, run_time