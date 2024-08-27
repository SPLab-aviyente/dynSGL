import numpy as np
import networkx as nx

def topo_und(G, n_swap=1, seed=None):
    """Perform edge swapping. The method randomly select two node pairs: (a, b) 
    and (c, d) where a, b, c, d are distinct from each other, (a, b) is an edge, 
    (c, d) is not. Then it removes (a, b, attr) where attr is the dictionary of 
    attributes of (a, b) and it adds (c, d, attr) to the graph. The algorithm 
    does not preserve degree distribution of the original graph.

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

def topo_und_degree_preserved(G, n_swap=1, seed=None):
    """Perform edge swapping while preserving degree distribution. The method
    randomly select two node pairs: (a, b) and (c, d) where a, b, c, d are 
    distinct from each other, (a, b) is an edge, (c, d) is an edge. Then it 
    removes both (a, b, attr_ab) and (c, d, attr_cd) where attr_ab and attr_cd
    are edge attributes. It adds (a, c, attr_ab) and (b, d, attr_cd) to the graph.
    
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

    max_attempt = round(m)

    for _ in range(n_swap):     

        # Select an edge to swap: ensures that a and b have at least two edges
        attempt = 0
        while attempt <= max_attempt:
            a = rng.choice(node_labels)

            if G.degree(a) > 0:
                b = rng.choice(list(G.neighbors(a)))
                break

        # Select two nodes to connect with an edge
        attempt = 0
        while attempt <= max_attempt:
            attempt += 1

            c = rng.choice(node_labels)

            if c == a or c == b:
                continue
        
            d = rng.choice(list(G.neighbors(c)))

            if d == a or d == b:
                continue

            break

        if attempt > max_attempt:
            continue
        
        # Rewire the edge
        attr_ab = G[a][b]
        attr_cd = G[c][d]
        G.remove_edge(a, b)
        G.remove_edge(c, d)

        G.add_edge(a, c, **attr_ab)
        G.add_edge(b, d, **attr_cd)
