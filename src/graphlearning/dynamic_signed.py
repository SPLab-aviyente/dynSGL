import numpy as np

from numba import njit
from scipy import sparse

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

def learn_a_dynamic_signed_graph(X, density, similarity, **kwargs):
    if not isinstance(X, list):
        raise Exception("Multiple sets of graph signals must be provided when "
                        "learning a dynamic signed graph.")

    n_times = len(X)
    n_nodes = X[0].shape[0]
    S = utils.rowsum_mat(n_nodes)

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

    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
    else:
        alpha = {"+": 0.1*np.ones(n_times), 
                 "-": 0.1*np.ones(n_times)}

    if "beta" in kwargs:
        beta = kwargs["beta"]
    else:
        beta = {"+": 0.01*np.ones(n_times-1), 
                "-": 0.01*np.ones(n_times-1)}

    rho = kwargs["rho"] if "rho" in kwargs else 10
    max_iter = kwargs["max_iter"] if "max_iter" in kwargs else 1000

    view_densities_ma = {"+": [], "-": []}
    similarities_ma = {"+": [], "-": []}

    rng = np.random.default_rng()

    iter = 0
    while True:
        iter += 1

        w = _run(data_vecs, alpha, beta, S, rho, max_iter)

        no_update = True
        view_densities = {"+": np.zeros(n_times), "-": np.zeros(n_times)}
        similarities = {"+": np.zeros(n_times-1), "-": np.zeros(n_times-1)}
        for s in ["+", "-"]:
            for t in range(n_times):
                density_hat = _density(w[s][t])
                view_densities[s][t] = density_hat

                # print(f"View {t}{s} density: {density_hat:.3f}")

                rnd = rng.uniform(0.8, 1.0)
                diff = density_hat - density
                if diff > 0.025:
                    alpha[s][t] = max(alpha[s][t] - 2*abs(diff), alpha[s][t]*(1-rnd*0.3))
                    no_update = False
                elif diff < -0.025:
                    alpha[s][t] = min(alpha[s][t] + 2*abs(diff), alpha[s][t]*(1+rnd*0.3))
                    no_update = False

                if t>0:
                    similarity_hat = _similarity(w[s][t], w[s][t-1])
                    similarities[s][t-1] = similarity_hat

                    # print(f"View {t}{s} similarity: {similarity_hat:.3f}")

                    diff = similarity_hat - similarity

                    if diff > 0.025:
                        beta[s][t-1] = max(beta[s][t-1] - 2*abs(diff), beta[s][t-1]*(1-rnd*0.3))
                        no_update = False
                    elif diff < -0.025:
                        beta[s][t-1] = min(beta[s][t-1] + 2*abs(diff), beta[s][t-1]*(1+rnd*0.3))
                        no_update = False

        # print("="*20)

        if no_update:
            break

        if iter > 6:
            change = lambda x, y: np.mean(np.abs(np.mean(x, axis=0) - y))
            if ((np.abs(change(similarities_ma["+"], similarities["+"])) < 1e-4) and
                (np.abs(change(view_densities_ma["+"], view_densities["+"])) < 1e-4) and 
                (np.abs(change(similarities_ma["-"], similarities["-"])) < 1e-4) and
                (np.abs(change(view_densities_ma["-"], view_densities["-"])) < 1e-4)):
                break

        if iter > 50:
            print("Hyperparameter search run too much. It is aborted.",
                  "Try initiating them at different values.")
            break

        for s in ["+", "-"]:
            view_densities_ma[s].append(view_densities[s].copy())
            similarities_ma[s].append(similarities[s].copy())
        
            if len(view_densities_ma[s]) > 5:
                view_densities_ma[s].pop(0)
            if len(similarities_ma[s]) > 5:
                similarities_ma[s].pop(0)

    # Optimal parameter values
    params = {
        "alpha": alpha,
        "beta": beta,
    }

    return w, params