import numpy as np

import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import copy
import jax
import jax.numpy as jnp
from scipy.linalg import solve as scipy_solve
import itertools
from  matplotlib.collections import LineCollection
from matplotlib.collections import EllipseCollection
import matplotlib.patheffects as path_effects
import matplotlib.tri as tri
from jax import jit

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

    def __init__(self, graph, jax=False):
        if type(graph) == str:
            self.graph = nx.read_gpickle(graph)
        else:
            self.graph = copy.deepcopy(graph)

        self.n = len(self.graph.nodes)
        self.ne = len(self.graph.edges)
        self.pts = np.array([self.graph.nodes[node]['pos'] for node in graph.nodes])

        self.incidence_matrix = nx.incidence_matrix(self.graph, oriented=True)
        if jax:
            self.incidence_matrix = jnp.array(self.incidence_matrix.todense())
            

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

    def jax_setConductances(self, conductances):
        # conductances is a list of floats
        assert len(conductances) == self.ne, 'conductances must have the same length as the number of edges'
        # if list, convert to numpy array
        if type(conductances) == list:
            conductances = jnp.array(conductances)
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

    def _jax_hessian(self):
        ''' Compute the Hessian of the network with respect to the conductances. '''
        return jnp.dot(self.incidence_matrix*self.conductances,jnp.transpose(self.incidence_matrix))
    
    @staticmethod
    def _shessian(conductances,incidence_matrix):
        ''' Compute the Hessian of the network with respect to the conductances. '''
        return jnp.dot(incidence_matrix*conductances,jnp.transpose(incidence_matrix))
    
    def constraint_matrix(self, indices_nodes, restrictionType = 'node'):
        ''' Compute the constraint matrix Q for the circuit and the nodes represented by indices_nodes. 
        Q is a sparse constraint rectangular matrix of size n x len(indices_nodes). Its entries are only 1 or 0.
        Q.Q^T is a projector onto to the space of the nodes.

        Parameters
		----------
		indices_nodes : np.array
            if restrictionType == 'node':
    			Array with the indices of the nodes to be constrained. The nodes themselves are given by np.array(self.graph.nodes)[indices_nodes].
            if restrictionType == 'edge':
                Array with the pairs of indices of the nodes to be constrained. The order of the elements signal the direction of the flow constraint,
                from the first to the second.

		Returns
		-------
		Q: scipy.sparse.csr_matrix
			Constraint matrix Q: a sparse constraint rectangular matrix of size n x len(indices_nodes). Its entries are only 1, -1, or 0.
            Q.Q^T is a projector onto to the space of the nodes.
		
        '''
        # Check indicesNodes is a non-empty array
        if len(indices_nodes) == 0:
            raise ValueError('indicesNodes must be a non-empty array.')
        if restrictionType == 'node':
            # check that indices_nodes has a valid shape (integer,)
            if len(indices_nodes.shape) != 1:
                raise ValueError('indices_nodes must be a 1D array.')
            # Create the sparse rectangular constraint matrix Q. Q has entries 1 at the indicesNodes[i] row and i column.
            Q = csr_matrix((np.ones(len(indices_nodes)), (indices_nodes, np.arange(len(indices_nodes)))), shape=(self.n, len(indices_nodes)))
        elif restrictionType == 'edge':
            # check that indices_nodes has a valid shape (integer, 2)
            if len(indices_nodes.shape) != 2 or indices_nodes.shape[1] != 2:
                raise ValueError('indices_nodes must be a 2D array with shape (integer, 2).')
            # Create the sparse rectangular constraint matrix Q. Q has entries 1 at the indicesNodes[i,0] row and i column, and -1 at the indicesNodes[i,1] row and i column.
            Q = csr_matrix((np.ones(len(indices_nodes)), (indices_nodes[:,0], np.arange(len(indices_nodes)))), shape=(self.n, len(indices_nodes))) + csr_matrix((-np.ones(len(indices_nodes)), (indices_nodes[:,1], np.arange(len(indices_nodes)))), shape=(self.n, len(indices_nodes)))
        else:
            raise ValueError('restrictionType must be either "node" or "edge".')
        return Q
    
    def jax_constraint_matrix(self, indices_nodes, restrictionType = 'node'):
        ''' Compute the constraint matrix Q for the circuit and the nodes represented by indices_nodes. 
        Q is a dense constraint rectangular matrix of size n x len(indices_nodes). Its entries are only 1 or 0.
        Q.Q^T is a projector onto to the space of the nodes.

        Parameters
		----------
		indices_nodes : np.array
            if restrictionType == 'node':
    			Array with the indices of the nodes to be constrained. The nodes themselves are given by np.array(self.graph.nodes)[indices_nodes].
            if restrictionType == 'edge':
                Array with the pairs of indices of the nodes to be constrained. The order of the elements signal the direction of the flow constraint,
                from the first to the second.
		Returns
		-------
		Q: 
			Constraint matrix Q: a dense constraint rectangular matrix of size n x len(indices_nodes). Its entries are only 1, -1, or 0.
            Q.Q^T is a projector onto to the space of the nodes.
		
        '''
        # # Check indicesNodes is a non-empty array
        # if len(indices_nodes) == 0:
        #     raise ValueError('indicesNodes must be a non-empty array.')
        # # Create the sparse rectangular constraint matrix Q. Q has entries 1 at the indicesNodes[i] row and i column.
        # Q = jnp.zeros(shape=(self.n, len(indices_nodes)))
        # Q = Q.at[indices_nodes, jnp.arange(len(indices_nodes))].set(1)
        # return Q


        # Check indicesNodes is a non-empty array
        if len(indices_nodes) == 0:
            raise ValueError('indicesNodes must be a non-empty array.')
        if restrictionType == 'node':
            # check that indices_nodes has a valid shape (integer,)
            if len(indices_nodes.shape) != 1:
                raise ValueError('indices_nodes must be a 1D array.')
            # Create the sparse rectangular constraint matrix Q. Q has entries 1 at the indicesNodes[i] row and i column.
            Q = jnp.zeros(shape=(self.n, len(indices_nodes)))
            Q = Q.at[indices_nodes, jnp.arange(len(indices_nodes))].set(1)
        elif restrictionType == 'edge':
            # check that indices_nodes has a valid shape (integer, 2)
            if len(indices_nodes.shape) != 2 or indices_nodes.shape[1] != 2:
                raise ValueError('indices_nodes must be a 2D array with shape (integer, 2).')
            # Create the sparse rectangular constraint matrix Q. Q has entries 1 at the indicesNodes[i,0] row and i column, and -1 at the indicesNodes[i,1] row and i column.
            Q = jnp.zeros(shape=(self.n, len(indices_nodes)))
            Q = Q.at[indices_nodes[:,0], jnp.arange(len(indices_nodes))].set(1)
            Q = Q.at[indices_nodes[:,1], jnp.arange(len(indices_nodes))].set(-1)
        else:
            raise ValueError('restrictionType must be either "node" or "edge".')
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

    def _jax_extended_hessian(self, Q):
        ''' Extend the hessian of the network with the constraint matrix Q. 

        Parameters
        ----------
        Q : 
            Constraint matrix Q

        Returns
        -------
        H : 
            Extended Hessian. H is a dense matrix of size (n + len(indices_nodes)) x (n + len(indices_nodes)).
        
        '''

        extendedHessian = jnp.block([[self._jax_hessian(), Q],[jnp.transpose(Q), jnp.zeros(shape=(jnp.shape(Q)[1],jnp.shape(Q)[1]))]])
        # bmat([[self._hessian(), Q], [Q.T, None]], format='csr', dtype=float)
        return extendedHessian

    @staticmethod
    def _sextended_hessian(hessian,Q):
        ''' Extend the hessian of the network with the constraint matrix Q. 

        Parameters
        ----------
        Q : 
            Constraint matrix Q

        Returns
        -------
        H : 
            Extended Hessian. H is a dense matrix of size (n + len(indices_nodes)) x (n + len(indices_nodes)).
        
        '''
        extendedHessian = jnp.block([[hessian, Q],[jnp.transpose(Q), jnp.zeros(shape=(jnp.shape(Q)[1],jnp.shape(Q)[1]))]])
        return extendedHessian

    


    
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

    def jax_solve(self, Q, f):
        ''' Solve the circuit with the constraint matrix Q and the source vector f.

        Parameters
        ----------
        Q : jnp.array
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
        H = self._jax_extended_hessian(Q)
        # extend f with n zeros
        f_extended = jnp.hstack([jnp.zeros(self.n), f])
        # solve the system
        V = jax.scipy.linalg.solve(H, f_extended)[:self.n]
        return V

    @staticmethod
    @jit
    def ssolve(conductances, incidence_matrix, Q, f):
        ''' Solve the circuit with the constraint matrix Q and the source vector f.

        Parameters
        ----------
        Q : jnp.array
            Constraint matrix Q
        f : np.array
            Source vector f. f has size len(indices_nodes).

        Returns
        -------
        x : np.array
            Solution vector V. V has size n.
        '''
        # extend the hessian
        H = Circuit._sextended_hessian(Circuit._shessian(conductances,incidence_matrix),Q)
        # extend f with n zeros
        f_extended = jnp.hstack([jnp.zeros(jnp.shape(incidence_matrix)[0]), f])
        # solve the system
        V = jax.scipy.linalg.solve(H, f_extended)[:jnp.shape(incidence_matrix)[0]]
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
            
    def _remove_edge(self, edges):
        ''' Remove the edge from the graph. '''
        # determine the index of the edge in the list of edges
        index_edges = []
        for edge in edges:
            index_edges.append(list(self.graph.edges).index(tuple(edge)))
        
        self.graph.remove_edges_from(edges)

        if len(list(nx.isolates(self.graph)))>0:

            if np.intersect1d(np.r_[self.indices_source, self.indices_target.flatten()], list(nx.isolates(self.graph))).size > 0:
                raise ValueError

            self.graph.remove_nodes_from(list(nx.isolates(self.graph)))
            idxSort = np.argsort(list(self.graph.nodes))
            relabelMapping = {list(self.graph.nodes)[i]:idxSort[i] for i in range(idxSort.shape[0])}
            self.graph = nx.relabel_nodes(self.graph, relabelMapping)
            
        else:
            relabelMapping = np.arange(self.n)

        # remove the corresponding conductance
        self.conductances = np.delete(self.conductances, index_edges)

        return relabelMapping
        
    '''
	*****************************************************************************************************
	*****************************************************************************************************

										EFFECTIVE CONDUCTANCES

	*****************************************************************************************************
	*****************************************************************************************************
	'''

    def power_dissipated(self, voltages):
        ''' Compute the power dissipated in the circuit for the given voltages. '''
        # check that the conductances have been set
        try:
            self.conductances
        except AttributeError:
            raise AttributeError('Conductances have not been set yet.')
        # check that the voltages have the right size
        if len(voltages) != self.n:
            raise ValueError('Voltages have the wrong size.')
        # compute the power dissipated
        voltage_drop = self.incidence_matrix.T.dot(voltages)
        return np.sum(self.conductances*(voltage_drop**2)/2)

    def effective_conductance(self, array_pair_indices_nodes, seed = 0):
        ''' 
        Compute the effective conductance between the pair of nodes represented contained in array_pair_indices_nodes.

        Parameters
        ----------
        array_pair_indices_nodes : np.array
            Array with the pairs of indices of the nodes for which we want the effective conductance. The order of the elements does not matter.
        seed : int, optional
            Seed for the random number generator. The default is 0.

        Returns
        -------
        voltage_matrix : np.array
            Matrix of size n_unique_nodes x n_unique_nodes. Each row corresponds to the (DV)^2/2 between all the different pairs of unique nodes in array_pair_indices_nodes.
        power_vector : np.array
            Vector of size n_unique_nodes. Each entry corresponds to the power dissipated in the circuit for the corresponding row in voltage_matrix.
        effective_conductance : np.array
            Vector of size len(array_pair_indices_nodes). Each entry corresponds to the effective conductance between the pair of nodes in array_pair_indices_nodes.
        '''
        # check that the conductances have been set
        try:
            self.conductances
        except AttributeError:
            raise AttributeError('Conductances have not been set yet.')
        # check that the indices_nodes are valid
        if array_pair_indices_nodes.shape[1] != 2:
            raise ValueError('array_pair_indices_nodes must be a 2D array with shape (integer, 2).')


        # extract the unique indices of the nodes
        indices_nodes = np.unique(array_pair_indices_nodes)
        # check that the nodes exist
        if not all([node in self.graph.nodes for node in indices_nodes]):
            raise ValueError('Some of the nodes do not exist.')

        # generate array of all possible permutations of the indices_nodes
        all_possible_pairs = np.array(list(itertools.combinations(indices_nodes, 2)))

        # sort the arrays for better performance
        sorted_array_pair_indices_nodes = np.sort(array_pair_indices_nodes, axis=1)
        sorted_all_possible_pairs = np.sort(all_possible_pairs, axis=1)

        matching_indices = []

        for pair in sorted_array_pair_indices_nodes:
            index = np.where(np.all(sorted_all_possible_pairs == pair, axis=1))[0]
            if index.size != 0:
                matching_indices.append(index[0])


        # generate the linear system
        voltage_matrix = []
        power_vector = []

        np.random.seed(seed)

        for i in range(len(all_possible_pairs)):
            f = np.random.rand(len(indices_nodes))
            # generate the constraint matrix
            Q = self.constraint_matrix(indices_nodes, restrictionType = 'node')
            # solve the system
            V_free = self.solve(Q, f)
            # compute the voltage drops square based on the free state voltages and the array_pair_indices_nodes
            voltage_drops = np.array([(V_free[all_possible_pairs[i,0]] - V_free[all_possible_pairs[i,1]])**2 for i in range(len(all_possible_pairs))])
            # compute the power dissipated
            power_dissipated = self.power_dissipated(V_free)
            # add the voltage drops to the voltage matrix. Same for power
            voltage_matrix.append(voltage_drops/2)
            power_vector.append(power_dissipated)

        # compute the effective conductance
        voltage_matrix = np.array(voltage_matrix)
        power_vector = np.array(power_vector)
        # solve the linear system
        effective_conductance = scipy_solve(voltage_matrix, power_vector)

        # find the effective conductance corresponding to the pair of nodes in array_pair_indices_nodes
        effective_conductance = effective_conductance[matching_indices]

        return voltage_matrix, power_vector, effective_conductance



    '''
	*****************************************************************************************************
	*****************************************************************************************************

										EIGENVALUES AND EIGENVECTORS

	*****************************************************************************************************
	*****************************************************************************************************
	'''



    '''
	*****************************************************************************************************
	*****************************************************************************************************

										PLOTTING AND ANIMATION

	*****************************************************************************************************
	*****************************************************************************************************
    '''

    def plot_node_state(self, node_state, title = None, lw = 0.5, cmap = 'RdYlBu_r', size_factor = 100, prop = True, figsize = (4,4), filename = None, ax = None):
        ''' Plot the state of the nodes in the graph.

        Parameters
        ----------
        node_state : np.array
            State of the nodes in the graph. node_state has size n.
        '''
        posX = self.pts[:,0]
        posY = self.pts[:,1]
        norm = plt.Normalize(vmin=np.min(node_state), vmax=np.max(node_state))
        if prop:
            size = size_factor*np.abs(node_state[:])
        else:   
            size = size_factor
        if ax is not None:
            ax.scatter(posX, posY, s = size, c = node_state[:],edgecolors = 'black',linewidth = lw,  cmap = cmap, norm = norm)
            ax.set( aspect='equal')
            # remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            # show the colorbar
            # ax.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.5)
            # set the title of each subplot to be the corresponding eigenvalue in scientific notation
            ax.set_title(title)
        else:
            fig, axs = plt.subplots(1,1, figsize = figsize, constrained_layout=True,sharey=True)
            axs.scatter(posX, posY, s = size, c = node_state[:],edgecolors = 'black',linewidth = lw,  cmap = cmap, norm = norm)
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

    def plot_edge_state(self, edge_state, title = None,lw = 0.5, cmap = 'YlOrBr', figsize = (4,4), minmax = None, filename = None, background_color = '0.75'):
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
        axs.set_facecolor(background_color)
        fig.set_facecolor(background_color)
        # remove frame
        for spine in axs.spines.values():
            spine.set_visible(False)
        # set the title of each subplot to be the corresponding eigenvalue in scientific notation
        axs.set_title(title)
        if filename:
            fig.savefig(filename, dpi = 300)

    def edge_state_to_ax(self, ax, edge_state, vmin, vmax, cmap = 'YlOrBr', lw = 1, annotate = False):
        '''
        Plot the state of the edges in the graph.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object where the plot will be drawn.
        edge_state : np.array
            State of the edges in the graph. edge_state has size ne.
        vmin : float
            Minimum value of the colormap.
        vmax : float
            Maximum value of the colormap.
        cmap : str, optional
            Colormap. The default is 'YlOrBr'.
        lw : float, optional
            Linewidth. The default is 1.
        annotate : bool, optional
            If True, the edges are annotated with their index. The default is False.

        Returns
        -------
        plt.cm.ScalarMappable
            ScalarMappable object that can be used to add a colorbar to the plot.
        '''
        _cmap = plt.cm.get_cmap(cmap)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        # create the line collection object
        pos_edges = [np.array([self.graph.nodes[edge[0]]['pos'], self.graph.nodes[edge[1]]['pos']]) for edge in self.graph.edges()]
        color_array = _cmap(norm(edge_state))
        lc = LineCollection(pos_edges, color = color_array, linewidths = lw, path_effects=[path_effects.Stroke(capstyle="round")])
        ax.add_collection(lc)

        if annotate:
            for i in range(self.ne):
                ax.annotate(str(i), (np.mean(pos_edges[i][:,0]), np.mean(pos_edges[i][:,1])), fontsize = 2*lw, color = 'black', ha = 'center', va = 'center', zorder = 3, path_effects=[path_effects.withStroke(linewidth=2*lw,
                                                        foreground="w")])

        return plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    def node_state_to_ax(self, ax, node_state, vmin, vmax, cmap = 'viridis', plot_mode = 'collection', radius = 0.1, zorder = 2, annotate = False):
        ''' Plot the state of the nodes in the graph.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object where the plot will be drawn.
        node_state : np.array
            State of the nodes in the graph. node_state has size n.
        vmin : float
            Minimum value of the colormap.
        vmax : float
            Maximum value of the colormap.
        cmap : str, optional
            Colormap. The default is 'RdYlBu_r'.
        plot_mode : str, optional
            If 'collection', the nodes are plotted as a collection plot. If 'triangulation', the nodes are plotted as a triangulation. The default is 'collection'.
        radius : float, optional
            Radius of the nodes. The default is 0.1.
        zorder : int, optional
            Zorder of the nodes. The default is 2.
        annotate : bool, optional
            If True, the nodes are annotated with their index. The default is False.

        Returns
        -------
        plt.cm.ScalarMappable
            ScalarMappable object that can be used to add a colorbar to the plot.
        '''
        posX = self.pts[:,0]
        posY = self.pts[:,1]
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        if plot_mode == 'collection':
            # create a collection of ellipses
            color_array = plt.cm.get_cmap(cmap)(norm(node_state))
            r = np.ones(self.n)*radius
            ec = EllipseCollection(r, r, np.zeros(self.n), color = color_array, linewidths = 1,offsets=self.pts,
                       offset_transform=ax.transData, zorder=zorder)
            ax.add_collection(ec)
            

            
        elif plot_mode == 'triangulation':
            triang = tri.Triangulation(posX, posY)
            ax.tricontourf(triang, node_state, cmap = cmap, norm = norm, levels = 100)

        if annotate:
            for i in range(self.n):
                ax.annotate(str(i), (posX[i], posY[i]), fontsize = 0.8*radius, color = 'black', ha = 'center', va = 'center', zorder = 3)

        # return the colorbar
        return plt.cm.ScalarMappable(norm=norm, cmap=cmap)
 