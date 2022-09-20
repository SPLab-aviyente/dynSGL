from locale import normalize
import warnings

import networkx as nx
import numpy as np

from scipy import linalg

warnings.simplefilter(action="ignore", category=FutureWarning)

def _signed_laplacians(G):
    A = nx.adjacency_matrix(G, weight="sign").toarray()
    
    Ap = A.copy()
    Ap[Ap<0] = 0
    Lp = np.diag(np.sum(Ap, axis=1)) - Ap
    
    An = A.copy()
    An[An>0] = 0
    An *= -1
    Ln = np.diag(np.sum(An, axis=1)) - An

    return Lp, Ln

def gen_signals_from_signed_graph(G, n_signals, filter="Gaussian", alpha=10, 
                                  noise_amount=0.1, seed=None):
    
    n_nodes = G.number_of_nodes()

    Lp, Ln = _signed_laplacians(G)
        
    # Get the graph Laplacian spectrum
    ep, Vp = linalg.eigh(Lp, overwrite_a=True)
    ep[ep < 1e-8] = 0

    en, Vn = linalg.eigh(Ln, overwrite_a=True)
    en[en < 1e-8] = 0

    # Filtering to generate smooth graph signals from X0
    if filter == "Gaussian":
        hp = np.zeros(n_nodes)
        hp[ep > 0] = 1/np.sqrt(ep[ep>0])
        hn = np.zeros(n_nodes)
        hn[en > 0] = np.sqrt(en[en>0])
    elif filter == "Tikhonov":
        hp = 1/(1+alpha*ep)
        hn = (1+alpha*en)
    elif filter == "Heat":
        hp = np.exp(-alpha*ep)
        hn = np.exp(alpha*en)

    mid = n_nodes//2
    hp[mid:] = 0
    hn[:mid] = 0

    hp /= np.linalg.norm(hp)
    hn /= np.linalg.norm(hn)

    # Generate white noise
    rng = np.random.default_rng(seed=seed)
    X0 = rng.multivariate_normal(np.zeros(n_nodes), np.eye(n_nodes), n_signals).T
    X0p = np.diag(hp)@Vp.T@X0
    X0n = np.diag(hn)@Vn.T@X0
    
    X = 0.5*(Vp@X0p + Vn@X0n)

    # Add noise
    rng = np.random.default_rng(seed=seed)
    X_norm = np.linalg.norm(X)
    E = rng.normal(0, 1, X.shape)
    E_norm = np.linalg.norm(E)
    X += E*(noise_amount*X_norm/E_norm)

    return X