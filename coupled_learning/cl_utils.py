import numpy as np
from .circuit_utils import Circuit
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from scipy.sparse import csr_matrix

class CL(Circuit):
    '''
    Class for coupled learning in linear circuits.

    Attributes
    ----------
    graph : networkx.Graph
        Graph of the circuit.
    n : int
        Number of nodes in the circuit.
    ne : int
        Number of edges in the circuit.
    pts : np.array
        Positions of the nodes.
    '''
    def __init__(self, graph, conductances, learning_rate=1.0, learning_step = 0, min_k = 1.e-6, max_k = 1.e6, name = 'CL'):
        ''' Initialize the coupled learning circuit.

        Parameters
        ----------
        graph : networkx.Graph
            Graph of the circuit.
        conductances : np.array
            Conductances of the edges of the circuit.
        '''
        super().__init__(graph)
        self.setConductances(conductances)
        self.learning_rate = learning_rate
        self.learning_step = learning_step
        self.epoch = 0
        self.min_k = min_k
        self.max_k = max_k
        self.losses = []
        self.end_epoch = []
        self.name = name
    
    def set_name(self, name):
        ''' Set the name of the learning circuit.

        Parameters
        ----------
        name : string
            Name of the circuit.
        '''
        self.name = name

    def _clip_conductances(self):
        ''' Clip the conductances to be between min_k and max_k.
        '''
        self.conductances = np.clip(self.conductances, self.min_k, self.max_k)

    def _add_source(self, indices_source, inputs_source):
        ''' Add source nodes and their inputs to the circuit.

        Parameters
        ----------
        indices_source : np.array
            Indices of the nodes of the source.
        inputs_source : np.array
            Inputs of the source.
        '''
        # check they have the same length
        assert len(indices_source) == len(inputs_source), 'indices_source and inputs_source must have the same length'
        self.indices_source = indices_source
        self.inputs_source = inputs_source

    def _add_target(self, indices_target, outputs_target, target_type):
        ''' Add target nodes and their outputs to the circuit.

        Parameters
        ----------
        indices_target : np.array
            Indices of the nodes of the target.
        outputs_target : np.array
            Outputs of the target.
        target_type : string
            target type, "node" or "edge"
        '''
        # check they have the same length
        assert len(indices_target) == len(outputs_target), 'indices_target and outputs_target must have the same length'
        self.indices_target = indices_target
        self.outputs_target = outputs_target
        # check that the target type is valid
        assert target_type in ['node', 'edge'], 'target_type must be "node" or "edge"'
        self.target_type = target_type
        

    def set_task(self, indices_source, inputs_source, indices_target, outputs_target, target_type='node'):
        ''' Set the task of the circuit.

        Parameters
        ----------
        indices_source : np.array
            Indices of the nodes of the source.
        inputs_source : np.array
            Inputs of the source.
        indices_target : np.array
            If target is node, indices of the nodes of the target.
            If target is edge, array with edge index, and nodes i, j. 
        outputs_target : np.array
            Outputs of the target.
        target_type : string
            target type, "node" or "edge"
        
        Returns
        -------
        Q_free : scipy.sparse.csr_matrix
            Constraint matrix Q_free: a sparse constraint rectangular matrix of size n x len(indices_source). Its entries are only 1 or 0.
        Q_clamped : scipy.sparse.csr_matrix
            Constraint matrix Q_clamped: a sparse constraint rectangular matrix of size n x (len(indices_source) + len(indices_target)). Its entries are only 1 or 0.
        '''
        self._add_source(indices_source, inputs_source)
        self._add_target(indices_target, outputs_target, target_type)
        # Compute the constraint matrices
        self.Q_free = self.constraint_matrix(self.indices_source)
        if self.target_type == 'node':
            self.Q_clamped = self.constraint_matrix(np.concatenate((self.indices_source, self.indices_target)))    
        elif self.target_type == 'edge':
            constraintClamped = np.zeros((self.n, self.indices_source.shape[0] + self.indices_target.shape[0]))
            constraintClamped[self.indices_source, np.arange(self.indices_source.shape[0])] = 1
            constraintClamped[self.indices_target[:,1], self.indices_source.shape[0]+ np.arange(self.indices_target.shape[0])] = 1
            constraintClamped[self.indices_target[:,2], self.indices_source.shape[0]+ np.arange(self.indices_target.shape[0])] = -1
            clampedStateConstraintMatrix = csr_matrix(constraintClamped)
            self.Q_clamped = clampedStateConstraintMatrix

        return self.Q_free, self.Q_clamped
    
    def get_free_state(self):
        ''' Return the free state of the circuit for the current task. '''
        # determine if a task was given
        if not hasattr(self, 'Q_free'):
            raise ValueError('No task was given to the circuit. Use set_task(indices_source, inputs_source, indices_target, outputs_target) to set the task.')
        return self.solve(self.Q_free, self.inputs_source)
    
    def get_power_state(self):
        ''' Return the power state of the circuit for the current task. '''
        free_state = self.get_free_state()
        voltage_drop_free = self.incidence_matrix.T.dot(free_state)
        return self.conductances*(voltage_drop_free**2)

        

    def _step_CL(self, eta = 0.001):
        ''' Perform a step of coupled learning. '''
        free_state = self.solve(self.Q_free, self.inputs_source)
        # nudge = free_state[self.indices_target] + eta*(self.outputs_target - free_state[self.indices_target])


        if self.target_type == 'node':
            nudge = free_state[self.indices_target] + eta * (self.outputs_target - free_state[self.indices_target])

        elif self.target_type == 'edge':

            DP = free_state[self.indices_target[:,1]] - free_state[self.indices_target[:,2]]
            nudge = DP + eta * (self.outputs_target - DP)

        clamped_state = self.solve(self.Q_clamped, np.concatenate((self.inputs_source, nudge)))

        # voltage drop
        voltage_drop_free = self.incidence_matrix.T.dot(free_state)
        voltage_drop_clamped = self.incidence_matrix.T.dot(clamped_state)

        # Update the conductances
        delta_conductances = -1.0/eta * (voltage_drop_clamped**2 - voltage_drop_free**2)
        self.conductances = self.conductances + self.learning_rate*delta_conductances
        self._clip_conductances()

        self.learning_step += 1
        
        return free_state, voltage_drop_free, delta_conductances, self.conductances
    
    def iterate_CL(self, n_steps, eta = 0.001):
        ''' Iterate coupled learning for n_steps. '''
        for i in range(n_steps):
            free_state, voltage_drop_free , delta_conductances , conductances = self._step_CL(eta)
        return free_state, voltage_drop_free , delta_conductances , conductances
    
    def MSE_loss(self, free_state):
        ''' Compute the MSE loss. '''
        if self.target_type == 'node':
            return 0.5*np.mean((free_state[self.indices_target] - self.outputs_target)**2)
        elif self.target_type == 'edge':
            freeState_DV = free_state[self.indices_target[:,1]] - free_state[self.indices_target[:,2]]
            return 0.5*np.mean((freeState_DV - self.outputs_target)**2)

    
    def train(self, n_epochs, n_steps_per_epoch, eta = 0.001, verbose = True, pbar = False, log_spaced = False, save_state = False, save_path = None):
        ''' Train the circuit for n_epochs. Each epoch consists of n_steps_per_epoch steps of coupled learning.
        If log_spaced is True, n_steps_per_epoch is overwritten and the number of steps per epoch is log-spaced, such that the total number of steps is n_steps_per_epoch * n_epochs.
        '''
        if pbar:
            epochs = tqdm(range(n_epochs))
        else:
            epochs = range(n_epochs)
        if log_spaced:
            n_steps = n_epochs * n_steps_per_epoch
            n_steps_per_epoch = log_partition(n_steps, n_epochs)
            for epoch in epochs:
                free_state, voltage_drop_free , delta_conductances , conductances = self.iterate_CL(n_steps_per_epoch[epoch], eta)
                self.losses.append(self.MSE_loss(free_state))
                if verbose:
                    print('Epoch: {}/{} | Loss: {}'.format(epoch,n_epochs-1, self.losses[-1]))
                self.epoch += 1
                self.end_epoch.append(self.learning_step)
                if save_state:
                    self.save(save_path)
            return self.losses, free_state, voltage_drop_free , delta_conductances , conductances
        else:
            for epoch in epochs:
                free_state, voltage_drop_free , delta_conductances , conductances = self.iterate_CL(n_steps_per_epoch, eta)
                self.losses.append(self.MSE_loss(free_state))
                if verbose:
                    print('Epoch: {}/{} | Loss: {}'.format(epoch,n_epochs-1, self.losses[-1]))
                self.epoch += 1
                self.end_epoch.append(self.learning_step)
                if save_state:
                    # save the state of the circuit after each epoch
                    self.save(save_path)
            return self.losses, free_state, voltage_drop_free , delta_conductances , conductances
    
    def save(self, path):
        ''' Save the circuit. '''
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def reset_training(self):
        ''' Reset the training. '''
        self.learning_step = 0
        self.epoch = 0
        self.end_epoch = []
        self.losses = []


    '''
	*****************************************************************************************************
	*****************************************************************************************************

										PLOTTING AND ANIMATION

	*****************************************************************************************************
	*****************************************************************************************************
    '''

    def plot_circuit(self, title=None, lw = 0.5, point_size = 100, highlight_nodes = True, figsize = (4,4), highlighted_point_size = 200, filename = None):
        ''' Plot the circuit.
        '''
        posX = self.pts[:,0]
        posY = self.pts[:,1]
        pos_edges = np.array([np.array([self.graph.nodes[edge[0]]['pos'], self.graph.nodes[edge[1]]['pos']]).T for edge in self.graph.edges()])
        fig, axs = plt.subplots(1,1, figsize = figsize, constrained_layout=True,sharey=True)
        for i in range(len(pos_edges)):
            axs.plot(pos_edges[i,0], pos_edges[i,1], c = 'black', lw = lw, zorder = 1)
        axs.scatter(posX, posY, s = point_size, c = 'black', zorder = 2)
        if highlight_nodes:
            axs.scatter(posX[self.indices_source], posY[self.indices_source], s = highlighted_point_size, c = 'red', zorder = 10)
            try:
                axs.scatter(posX[self.indices_target[:,1:]], posY[self.indices_target[:,1:]], s = highlighted_point_size, c = 'blue', zorder = 10)
            except:
                axs.scatter(posX[self.indices_target], posY[self.indices_target], s = highlighted_point_size, c = 'blue', zorder = 10)
        axs.set( aspect='equal')
        # remove ticks
        axs.set_xticks([])
        axs.set_yticks([])
        # set the title of each subplot to be the corresponding eigenvalue in scientific notation
        axs.set_title(title)
        if filename:
            fig.savefig(filename, dpi = 300)





def load_circuit(path):
    ''' Load a circuit. '''
    with open(path, 'rb') as f:
        circuit = pickle.load(f)
    return circuit



def log_partition(N, M):
    if M > N:
        raise ValueError('M cannot be larger than N')
    # Create a logarithmically spaced array between 0 and N
    log_space = np.logspace(0, np.log10(N), M, endpoint=True, base=10.0)
    # Round the elements to the nearest integers
    log_space = np.rint(log_space).astype(int)
    # Calculate the differences between consecutive elements
    intervals = np.diff(log_space)
    # Ensure that no interval has a size of 0
    for i in range(len(intervals)):
        if intervals[i] == 0:
            intervals[i] += 1
    # Append N (the last element of the sequence) to the intervals
    intervals = np.append(intervals, N)
    # Adjust the intervals until they add up to N
    while np.sum(intervals) < N:
        intervals[np.argmin(intervals)] += 1
    while np.sum(intervals) > N:
        intervals[np.argmax(intervals)] -= 1
    return intervals.tolist()