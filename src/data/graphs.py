import networkx as nx

def gen_er_graph(n_nodes, edge_prob, seed=None, max_iter=100):
    """Generate an Erdos-Renyi graph whose connectedness is guaranteed. 
    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    edge_prob : float
        Edge probability
    seed : int, optional
        Seed for the random number generator, by default None
    max_iter : int, optional
        Maximum number of iterations to try to generate a connected graph, by 
        default 100
    Returns
    -------
    G : nx.graph
        Generated graph.
    Raises
    ------
    Exception
        When maximum number of iterations are reached.
    """

    iter = 0
    while True:
        G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed)

        if nx.is_connected(G):
            break

        if seed:
            seed += 1

        iter += 1

        if iter >= max_iter:
            raise Exception(("Cannot create a connected graph, "
                             "please increase edge probability."))
    
    return G

def gen_ba_graph(n_nodes, n_edges, seed=None, max_iter=100):
    """Generate a Barabasi-Albert graph whose connectedness is guaranteed
    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    n_edges : int
        Number of edges to attach at each step of graph generation.
    seed : int, optional
        Seed for the random number generator, by default None
    max_iter : int, optional
        Maximum number of iterations to try to generate a connected graph, by 
        default 100
    Returns
    -------
    G : nx.graph
        Generated graph.
    Raises
    ------
    Exception
        When maximum number of iterations are reached.
    """
    iter = 0
    while True:
        G = nx.barabasi_albert_graph(n_nodes, n_edges, seed)

        if nx.is_connected(G):
            break

        if seed:
            seed += 1

        iter += 1

        if iter >= max_iter:
            raise Exception(("Cannot create a connected graph, "
                             "please increase number of edges to be attached."))
    
    return G