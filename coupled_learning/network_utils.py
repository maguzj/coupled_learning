import networkx as nx
import numpy as np
import json
from packing_utils import *

def grid_network(m, n, periodic=False, size_uc = (1,1), relabel_nodes=True):
    graph = nx.grid_2d_graph(m, n, periodic=periodic)
    for node in graph.nodes:
        graph.nodes[node]['pos'] = np.array(node)*size_uc
    if relabel_nodes:
        graph = nx.convert_node_labels_to_integers(graph)

    return graph

def jammed_network(n, seed=0, relabel_nodes=True, rfac = 0.8, radius=0, params={'central':0.0005, 'drag':0.05, 'contact':0.1}):
    net = Packing(n, dim=2, radius=radius, rfac=rfac, params=params, seed=seed)
    # net.params['contact'] = 0.2 # reduce the default contact repulsion
    net.generate()
    graph = net.graph
    for node in graph.nodes:
        graph.nodes[node]['pos'] = graph.nodes[node]['pos'][:2]/net.radius
    if relabel_nodes:
        graph = nx.convert_node_labels_to_integers(graph)
    
    return graph

def network_from_json(filename):
    with open(filename) as f:
        data = json.load(f)
        nodes = data['nodes']
        edges = data['edges']
        pts = data['pts']

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        for i, node in enumerate(graph.nodes):
            graph.nodes[node]['pos'] = np.array(pts[i])

        return graph

# def network_to_json(graph, filename):
#     with open(filename, 'w') as f:
#         json.dump({
#             "nodes":list(graph.nodes),
#             "pts":pts.tolist(),
#             "edges":list(graph.edges)},f)