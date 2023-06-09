import numpy as np

import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class Circuit(object):
    ''' Class to simulate a circuit with trainable conductances 
    
    Parameters
    ----------
    graph   :   str or networkx.Graph
                If str, it is the path to the file containing the graph. If networkx.Graph, it is the graph itself.
    
    Attributes
    ----------
    graph   :   networkx.Graph
                Graph specifying the nodes and edges in the network. A conductance parameter is associated with each edge. A trainable edge will
		        be updated during training.
    n : int
        Number of nodes in the graph.
    ne : int
        Number of edges in the graph.
    pts: numpy.ndarray
        Positions of the nodes in the graph.
    '''

    def __init__(self, graph):
        if type(graph) == str:
            self.graph = nx.read_gpickle(graph)
        else:
            self.graph = graph

        self.n = len(self.graph.nodes)
        self.ne = len(self.graph.edges)
        self.pts = np.array([self.graph.nodes[node]['pos'] for node in graph.nodes])
        
        self.incidence_matrix = nx.incidence_matrix(self.graph, oriented=True)

    # def setPositions(self, positions):
    #     # positions is a list of tuples
    #     assert len(positions) == self.n, 'positions must have the same length as the number of nodes'
    #     self.pts = positions

    def setConductances(self, conductances):
        # conductances is a list of floats
        assert len(conductances) == self.ne, 'conductances must have the same length as the number of edges'
        # if list, convert to numpy array
        if type(conductances) == list:
            conductances = np.array(conductances)
        self.conductances = conductances

    def _setConductances_as_weights(self):
        ''' Set the conductances as weights of the edges in the graph. '''
        nx.set_edge_attributes(self.graph, dict(zip(self.graph.edges, self.conductances)), 'weight')

    def weight_to_conductance(self):
        ''' Set the conductances as weights of the edges in the graph. '''
        self.conductances = np.array([self.graph.edges[edge]['weight'] for edge in self.graph.edges])

    def _hessian(self):
        ''' Compute the Hessian of the network with respect to the conductances. '''
        return (self.incidence_matrix*self.conductances).dot(self.incidence_matrix.T)
    
    def constraint_matrix(self, indices_nodes):
        ''' Compute the constraint matrix Q for the circuit and the nodes represented by indices_nodes. 
        Q is a sparse constraint rectangular matrix of size n x len(indices_nodes). Its entries are only 1 or 0.
        Q.Q^T is a projector onto to the space of the nodes.

        Parameters
		----------
		indices_nodes : np.array
			Array with the indices of the nodes to be constrained. The nodes themselves are given by np.array(self.graph.nodes)[indices_nodes].

		Returns
		-------
		Q: scipy.sparse.csr_matrix
			Constraint matrix Q: a sparse constraint rectangular matrix of size n x len(indices_nodes). Its entries are only 1 or 0.
            Q.Q^T is a projector onto to the space of the nodes.
		
        '''
         # Check indicesNodes is a non-empty array
        if len(indices_nodes) == 0:
            raise ValueError('indicesNodes must be a non-empty array.')
        # Create the sparse rectangular constraint matrix Q using csr_matrix. Q has entries 1 at the indicesNodes[i] row and i column.
        Q = csr_matrix((np.ones(len(indices_nodes)), (indices_nodes, np.arange(len(indices_nodes)))), shape=(self.n, len(indices_nodes)))
        return Q
    
    def _extended_hessian(self, Q):
        ''' Extend the hessian of the network with the constraint matrix Q. 

        Parameters
        ----------
        Q : scipy.sparse.csr_matrix
            Constraint matrix Q

        Returns
        -------
        H : scipy.sparse.csr_matrix
            Extended Hessian. H is a sparse matrix of size (n + len(indices_nodes)) x (n + len(indices_nodes)).
        
        '''
        sparseExtendedHessian = bmat([[self._hessian(), Q], [Q.T, None]], format='csr', dtype=float)
        return sparseExtendedHessian


    
    '''
	*****************************************************************************************************
	*****************************************************************************************************

										NUMERICAL INTEGRATION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

    def solve(self, Q, f):
        ''' Solve the circuit with the constraint matrix Q and the source vector f.

        Parameters
        ----------
        Q : scipy.sparse.csr_matrix
            Constraint matrix Q
        f : np.array
            Source vector f. f has size len(indices_nodes).

        Returns
        -------
        x : np.array
            Solution vector V. V has size n.
        '''
        # check that the conductances have been set
        try:
            self.conductances
        except AttributeError:
            raise AttributeError('Conductances have not been set yet.')
        # check that the source vector has the right size
        if len(f) != Q.shape[1]:
            raise ValueError('Source vector f has the wrong size.')
        # extend the hessian
        H = self._extended_hessian(Q)
        # extend f with n zeros
        f_extended = np.hstack([np.zeros(self.n), f])
        # solve the system
        V = spsolve(H, f_extended)[:self.n]
        return V


    '''
	*****************************************************************************************************
	*****************************************************************************************************

										CIRCUIT REDUCTION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

    def _star_to_mesh_onenode(self, node):
        ''' Reduces the circuit by using the star-mesh transformation on node.

        Parameters
        ----------
        node : int
            Index of the node to be transformed.

        '''
        # determine the neighbors of node and the sum of the conductances of the edges between node and its neighbors
        neighbors = list(self.graph.neighbors(node))
        sum_conductances = sum([self.graph[node][neighbor]['weight'] for neighbor in neighbors])

        # add the edges between the neighbors of node with new weights (conductances) corresponding to the product of the conductances of the edges between node and its neighbors divided by the sum of the conductances of the edges between node and its neighbors
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                eff_conductance = (self.graph[neighbors[i]][node]['weight'])*(self.graph[neighbors[j]][node]['weight'])/sum_conductances
                # add the edge if it does not exist
                if not self.graph.has_edge(neighbors[i], neighbors[j]):
                    self.graph.add_edge(neighbors[i], neighbors[j], weight = eff_conductance)
                # otherwise, update the weight (parallel conductances)
                else:
                    self.graph[neighbors[i]][neighbors[j]]['weight'] += eff_conductance

        # remove the node from the graph
        self.graph.remove_node(node)

    def star_to_mesh(self, nodes, all_but_nodes=False):
        ''' Reduces the circuit by using the star-mesh transformation on all nodes in nodes.
        
        Parameters
        ----------
        nodes : np.array
            Array with the indices of the nodes to be transformed.
        all_but_nodes : bool, optional
            If True, the star-mesh transformation is applied to all nodes except those in nodes. The default is False.
        '''
        # check that nodes is a non-empty array
        if len(nodes) == 0:
            raise ValueError('nodes must be a non-empty array.')
        # check that the nodes exist
        if not all([node in self.graph.nodes for node in nodes]):
            raise ValueError('Some of the nodes do not exist.')
        # check that the nodes are not repeated
        if len(nodes) != len(set(nodes)):
            raise ValueError('Some of the nodes are repeated.')
        # check that the graph will have at least 2 nodes
        if (len(nodes) >= self.n - 1 and not all_but_nodes) or (len(nodes) < 2 and all_but_nodes):
            raise ValueError('The graph will have less than 2 nodes.')

        if all_but_nodes:
            nodes = [node for node in self.graph.nodes() if node not in nodes]

        # apply the star-mesh transformation to all nodes in nodes
        for node in nodes:
            self._star_to_mesh_onenode(node)

        self.n = self.graph.number_of_nodes()
        self.ne = self.graph.number_of_edges()
        self.pts = np.array([self.graph.nodes[node]['pos'] for node in self.graph.nodes])
        self.weight_to_conductance()
            
        


        
    



    '''
	*****************************************************************************************************
	*****************************************************************************************************

										PLOTTING AND ANIMATION

	*****************************************************************************************************
	*****************************************************************************************************
    '''

    def plot_node_state(self, node_state, title = None, lw = 0.5, cmap = 'RdYlBu_r', size_factor = 100, figsize = (4,4), filename = None):
        ''' Plot the state of the nodes in the graph.

        Parameters
        ----------
        node_state : np.array
            State of the nodes in the graph. node_state has size n.
        '''
        posX = self.pts[:,0]
        posY = self.pts[:,1]
        norm = plt.Normalize(vmin=np.min(node_state), vmax=np.max(node_state))
        fig, axs = plt.subplots(1,1, figsize = figsize, constrained_layout=True,sharey=True)
        axs.scatter(posX, posY, s = size_factor*np.abs(node_state[:]), c = node_state[:],edgecolors = 'black',linewidth = lw,  cmap = cmap, norm = norm)
        axs.set( aspect='equal')
        # remove ticks
        axs.set_xticks([])
        axs.set_yticks([])
        # show the colorbar
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, shrink=0.5)
        # set the title of each subplot to be the corresponding eigenvalue in scientific notation
        axs.set_title(title)
        if filename is not None:
            fig.savefig(filename, dpi = 300, bbox_inches='tight')

    def plot_edge_state(self, edge_state, title = None,lw = 0.5, cmap = 'RdYlBu_r', figsize = (4,4), minmax = None, filename = None):
        ''' Plot the state of the edges in the graph.

        Parameters
        ----------
        edge_state : np.array
            State of the edges in the graph. edge_state has size ne.
        '''
        _cmap = plt.cm.get_cmap(cmap)
        pos_edges = np.array([np.array([self.graph.nodes[edge[0]]['pos'], self.graph.nodes[edge[1]]['pos']]).T for edge in self.graph.edges()])
        if minmax:
            norm = plt.Normalize(vmin=minmax[0], vmax=minmax[1])
        else:
            norm = plt.Normalize(vmin=np.min(edge_state), vmax=np.max(edge_state))
        fig, axs = plt.subplots(1,1, figsize = figsize, constrained_layout=True,sharey=True)
        for i in range(len(pos_edges)):
            axs.plot(pos_edges[i,0], pos_edges[i,1], color = _cmap(norm(edge_state[i])), linewidth = lw)
        axs.set( aspect='equal')
        # remove ticks
        axs.set_xticks([])
        axs.set_yticks([])
        # show the colorbar
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, shrink=0.5)
        # set the title of each subplot to be the corresponding eigenvalue in scientific notation
        axs.set_title(title)
        if filename:
            fig.savefig(filename, dpi = 300)

 