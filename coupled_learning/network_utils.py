import networkx as nx
import numpy as np

def grid_network(m, n, periodic=False, size_uc = (1,1)):
    graph = nx.grid_2d_graph(m, n, periodic=periodic)
    for node in graph.nodes:
        graph.nodes[node]['pos'] = np.array(node)*size_uc

    return graph
