import numpy as np
import networkx as nx

def topological_undirected(G, n_swap=1, seed=None):
    """Perform edge swapping while preserving degree distribution. The method
    randomly select two node pairs: (a, b) and (c, d) where a, b, c, d are 
    distinct from each other, (a, b) is an edge, (c, d) is not. Then it removes 
    (a, b, attr) where attr is the dictionary of attributes of (a, b) and it 
    adds (c, d, attr) to the graph. The algorithm does not preserve degree 
    distribution of the original graph

    Parameters
    ----------
    G : networkx graph
        An undirected graph
    n_swap : int, optional
        Number of edge swap to perform, by default 1
    seed : int, optional
        Seed for the random number generator, by default None
    """

    n = G.number_of_nodes()
    m = G.number_of_edges()

    # If the graph is fully connected, there is nothing to randomize
    if m == (n*(n-1)/2):
        return
    
    # Get the nodes of the graph as a list
    node_labels = [i for i in G.nodes]

    rng = np.random.default_rng(seed=seed)

    max_attempt = round(n*m/(n*(n-1)))
    
    for _ in range(n_swap):     

        # Select an edge to swap: ensures that a and b have at least two edges
        attempt = 0
        while attempt <= max_attempt:
            a = rng.choice(node_labels)

            if G.degree(a) > 1:
                b = rng.choice(list(G.neighbors(a)))
            
                if G.degree(b) > 1:
                    break


        # Select two nodes to connect with an edge
        attempt = 0
        while attempt <= max_attempt:
            c = rng.choice(node_labels)
            d = rng.choice(node_labels)

            if c != b and c != a and d!=a and d != b and (not G.has_edge(c, d)):
                break
                
            attempt += 1
        
        # Rewire the edge
        attr = G[a][b]
        G.remove_edge(a, b)
        G.add_edge(c, d, **attr)