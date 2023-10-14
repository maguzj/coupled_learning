import numpy as np
# from .circuit_utils import Circuit
from circuit_utils import Circuit
from network_utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from scipy.sparse import csr_matrix
import jax.numpy as jnp
import jax
import json
import csv

from scipy.sparse import hstack

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
    def __init__(self, graph, conductances, learning_rate=1.0, learning_step = 0, min_k = 1.e-6, max_k = 1.e6, name = 'CL', jax = False, losses = None, end_epoch = None):
        ''' Initialize the coupled learning circuit.

        Parameters
        ----------
        graph : networkx.Graph
            Graph of the circuit.
        conductances : np.array
            Conductances of the edges of the circuit.
        '''
        super().__init__(graph, jax=jax)
        self.jax = jax
        self.setConductances(conductances)
        self.learning_rate = learning_rate
        self.learning_step = learning_step
        
        self.min_k = min_k
        self.max_k = max_k
        if losses is None:
            self.losses = []
        else:
            self.losses = losses
        if end_epoch is None:
            self.end_epoch = []
            self.epoch = 0
        else:
            self.epoch = end_epoch[-1]
            self.end_epoch = end_epoch
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

    def _jax_clip_conductances(self):
        ''' Clip the conductances to be between min_k and max_k.
        '''
        self.conductances = jnp.clip(self.conductances, self.min_k, self.max_k)

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
            q_edge = self.constraint_matrix(indices_target, restrictionType='edge')
            self.Q_clamped = hstack([self.Q_free, q_edge])
            # constraintClamped = np.zeros((self.n, self.indices_source.shape[0] + self.indices_target.shape[0]))
            # constraintClamped[self.indices_source, np.arange(self.indices_source.shape[0])] = 1
            # constraintClamped[self.indices_target[:,1], self.indices_source.shape[0]+ np.arange(self.indices_target.shape[0])] = 1
            # constraintClamped[self.indices_target[:,2], self.indices_source.shape[0]+ np.arange(self.indices_target.shape[0])] = -1
            # clampedStateConstraintMatrix = csr_matrix(constraintClamped)
            # self.Q_clamped = clampedStateConstraintMatrix

        return self.Q_free, self.Q_clamped

    def jax_set_task(self, indices_source, inputs_source, indices_target, outputs_target, target_type='node'):
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
        self.Q_free = self.jax_constraint_matrix(self.indices_source)
        if self.target_type == 'node':
            self.Q_clamped = self.jax_constraint_matrix(jnp.concatenate((self.indices_source, self.indices_target)))    
        elif self.target_type == 'edge':
            # constraintClamped = np.zeros((self.n, self.indices_source.shape[0] + self.indices_target.shape[0]))
            # constraintClamped[self.indices_source, np.arange(self.indices_source.shape[0])] = 1
            # constraintClamped[self.indices_target[:,1], self.indices_source.shape[0]+ np.arange(self.indices_target.shape[0])] = 1
            # constraintClamped[self.indices_target[:,2], self.indices_source.shape[0]+ np.arange(self.indices_target.shape[0])] = -1
            # clampedStateConstraintMatrix = csr_matrix(constraintClamped)
            # self.Q_clamped = clampedStateConstraintMatrix
            # Throw exception
            raise Exception('jax_set_task not implemented for edge target type')

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

            # DP = free_state[self.indices_target[:,1]] - free_state[self.indices_target[:,2]]
            DP = free_state[self.indices_target[:,0]] - free_state[self.indices_target[:,1]]
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

    def _step_GD(self):
        ''' Perform a step of gradient descent over the MSE. '''
        # delta_conductances = -jax.grad(self.jax_MSE_loss)(self.conductances)
        self.conductances = self.conductances - self.learning_rate*jax.grad(self.jax_MSE_loss)(self.conductances)
        self.conductances = jnp.clip(self.conductances, self.min_k, self.max_k)
        self.learning_step += 1

        return self.conductances
        

    def iterate_CL(self, n_steps, eta = 0.001):
        ''' Iterate coupled learning for n_steps. '''
        for i in range(n_steps):
            free_state, voltage_drop_free , delta_conductances , conductances = self._step_CL(eta)
        return free_state, voltage_drop_free , delta_conductances , conductances

    def iterate_GD(self, n_steps):
        ''' Iterate gradient descent for n_steps. '''
        for i in range(n_steps):
            conductances = self._step_GD()
        return conductances
    
    def MSE_loss(self, free_state):
        ''' Compute the MSE loss. '''
        if self.target_type == 'node':
            return 0.5*np.mean((free_state[self.indices_target] - self.outputs_target)**2)
        elif self.target_type == 'edge':
            # freeState_DV = free_state[self.indices_target[:,1]] - free_state[self.indices_target[:,2]]
            freeState_DV = free_state[self.indices_target[:,0]] - free_state[self.indices_target[:,1]]
            return 0.5*np.mean((freeState_DV - self.outputs_target)**2)

    def jax_MSE_loss(self, conductances):
        ''' Compute the MSE loss. '''
        self.setConductances(conductances)
        free_state = self.jax_solve(self.Q_free, self.inputs_source)
        if self.target_type == 'node':
            return 0.5*jnp.mean((free_state[self.indices_target] - self.outputs_target)**2)
        elif self.target_type == 'edge':
            freeState_DV = free_state[self.indices_target[:,1]] - free_state[self.indices_target[:,2]]
            return 0.5*jnp.mean((freeState_DV - self.outputs_target)**2)

    
    def train(self, n_epochs, n_steps_per_epoch, eta = 0.001, verbose = True, pbar = False, log_spaced = False, save_global = False, save_state = False, save_path = None):
        ''' Train the circuit for n_epochs. Each epoch consists of n_steps_per_epoch steps of coupled learning.
        If log_spaced is True, n_steps_per_epoch is overwritten and the number of steps per epoch is log-spaced, such that the total number of steps is n_steps_per_epoch * n_epochs.
        '''
        if pbar:
            epochs = tqdm(range(n_epochs))
        else:
            epochs = range(n_epochs)
        # # save attributes
        # if save_path:
        #     self.attributes_to_file(save_path+'_attributes.txt')

        # initial state
        if self.learning_step == 0:
            self.end_epoch.append(self.learning_step)
            self.losses.append(self.MSE_loss(self.get_free_state()))
            if save_state:
                self.save_local(save_path+'.csv')

        #training
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
                    # self.save(save_path+'_epoch_'+str(epoch)+'.pkl')
                    self.save_local(save_path+'.csv')
            # at the end of training, save global and save graph
            if save_global:
                self.save_global(save_path+'_global.json')
                self.save_graph(save_path+'_graph.json')
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
                    # self.save(save_path+'_epoch_'+str(epoch)+'.pkl')
                    self.save_local(save_path+'.csv')
            # at the end of training, save global and save graph
            if save_global:
                self.save_global(save_path+'_global.json')
                self.save_graph(save_path+'_graph.json')
            return self.losses, free_state, voltage_drop_free , delta_conductances , conductances

    def train_GD(self, n_epochs, n_steps_per_epoch, verbose = True, pbar = False, log_spaced = False, save_state = False, save_path = None):
        ''' Train the circuit for n_epochs. Each epoch consists of n_steps_per_epoch steps of gradient descent.
        If log_spaced is True, n_steps_per_epoch is overwritten and the number of steps per epoch is log-spaced, such that the total number of steps is n_steps_per_epoch * n_epochs.
        '''
        if pbar:
            epochs = tqdm(range(n_epochs))
        else:
            epochs = range(n_epochs)
        # save attributes
        # if save_path:
        #     self.attributes_to_file(save_path+'_attributes.txt')

        # initial state
        if self.learning_step == 0:
            self.end_epoch.append(self.learning_step)
            self.losses.append(float(self.jax_MSE_loss(self.conductances)))
            if save_state:
                self.save_local(save_path+'.csv')

        #training

        if log_spaced:
            n_steps = n_epochs * n_steps_per_epoch
            n_steps_per_epoch = log_partition(n_steps, n_epochs)
            for epoch in epochs:
                conductances = self.iterate_GD(n_steps_per_epoch[epoch])
                self.losses.append(float(self.jax_MSE_loss(conductances)))
                if verbose:
                    print('Epoch: {}/{} | Loss: {}'.format(epoch,n_epochs-1, self.losses[-1]))
                self.epoch += 1
                self.end_epoch.append(self.learning_step)
                if save_state:
                    # self.save(save_path+'_epoch_'+str(epoch)+'.pkl')
                    self.save_local(save_path+'.csv')
            # at the end of training, save global and save graph
            if save_global:
                self.save_global(save_path+'_global.json')
                self.save_graph(save_path+'_graph.json')
            return self.losses, conductances
        else:
            for epoch in epochs:
                conductances = self.iterate_GD(n_steps_per_epoch)
                self.losses.append(float(self.jax_MSE_loss(conductances)))
                if verbose:
                    print('Epoch: {}/{} | Loss: {}'.format(epoch,n_epochs-1, self.losses[-1]))
                self.epoch += 1
                self.end_epoch.append(self.learning_step)
                if save_state:
                    # save the state of the circuit after each epoch
                    # self.save(save_path+'_epoch_'+str(epoch)+'.pkl')
                    self.save_local(save_path+'.csv')
            # at the end of training, save global and save graph
            if save_global:
                self.save_global(save_path+'_global.json')
                self.save_graph(save_path+'_graph.json')
            return self.losses, conductances
    
    # def save(self, path):
    #     ''' Save the circuit. '''
    #     with open(path, 'wb') as f:
    #         pickle.dump(self, f)

    def reset_training(self):
        ''' Reset the training. '''
        self.learning_step = 0
        self.epoch = 0
        self.end_epoch = []
        self.losses = []

    # def attributes_to_file(self, path):
    #     ''' Save the attributes of the circuit to a file. '''
    #     with open(path, 'w') as f:
    #         f.write('n: {}\n'.format(self.n))
    #         f.write('ne: {}\n'.format(self.ne))
    #         # f.write('graph: {}\n'.format(self.graph))
    #         # f.write('pts: {}\n'.format(self.pts))
    #         # f.write('conductances: {}\n'.format(self.conductances))
    #         f.write('learning_rate: {}\n'.format(self.learning_rate))
    #         f.write('learning_step: {}\n'.format(self.learning_step))
    #         f.write('epoch: {}\n'.format(self.epoch))
    #         f.write('min_k: {}\n'.format(self.min_k))
    #         f.write('max_k: {}\n'.format(self.max_k))
    #         # f.write('losses: {}\n'.format(self.losses))
    #         f.write('end_epoch: {}\n'.format(self.end_epoch))
    #         f.write('name: {}\n'.format(self.name))
    #         f.write('indices_source: {}\n'.format(self.indices_source))
    #         f.write('inputs_source: {}\n'.format(self.inputs_source))
    #         f.write('indices_target: {}\n'.format(self.indices_target))
    #         f.write('outputs_target: {}\n'.format(self.outputs_target))
    #         f.write('target_type: {}\n'.format(self.target_type))
    #         # f.write('Q_free: {}\n'.format(self.Q_free))
    #         # f.write('Q_clamped: {}\n'.format(self.Q_clamped))

    '''
	*****************************************************************************************************
	*****************************************************************************************************

										SAVE AND EXPORT

	*****************************************************************************************************
	*****************************************************************************************************
    '''

    def save_graph(self, path):
        ''' Save the graph of the circuit in JSON format '''
        # first save the nodes ids, and their positions, then save the edges
        # with open(path, 'w') as f:
        #     f.write('{\n')
        #     f.write('\t"nodes": {},\n'.format(list(self.graph.nodes)))
        #     f.write('\t"pts": {},\n'.format(self.pts.tolist()))
        #     f.write('\t"edges": {}\n'.format(list(self.graph.edges)))
        #     f.write('}')

        with open(path, 'w') as f:
            json.dump({
                "nodes":list(self.graph.nodes),
                "pts":self.pts.tolist(),
                "edges":list(self.graph.edges)},f)

    def read_graph(self,path):
        ''' Read the graph of the circuit from JSON format '''
        with open(path, 'r') as f:
            data = json.load(f)
        # first read the nodes ids, and their positions, then read the edges
        self.graph = nx.Graph()
        self.graph.add_nodes_from(data['nodes'])
        self.pts = np.array(data['pts'])
        self.graph.add_edges_from(data['edges'])
        self.n = self.graph.number_of_nodes()
        self.ne = self.graph.number_of_edges()
        self.incidence_matrix = self.get_incidence_matrix()

    def save_global(self, path):
        ''' Save the attributes of the circuit in JSON format. '''
        # create a dictionary with the attributes
        dic = {
            "name": self.name,
            "n": self.n,
            "ne": self.ne,
            "learning_rate": self.learning_rate,
            "learning_step": self.learning_step,
            "epoch": self.epoch,
            "jax": self.jax,
            "min_k": self.min_k,
            "max_k": self.max_k,
            "indices_source": self.indices_source.tolist(),
            "inputs_source": self.inputs_source.tolist(),
            "indices_target": self.indices_target.tolist(),
            "outputs_target": self.outputs_target.tolist(),
            "target_type": self.target_type,
            "losses": self.losses,
            "end_epoch": self.end_epoch
        }
        # save the dictionary in JSON format
        with open(path, 'w') as f:
            json.dump(dic, f)

        # with open(path, 'w') as f:
        #     f.write('{\n')
        #     f.write('\t"name": "{}",\n'.format(self.name))
        #     f.write('\t"n": {},\n'.format(self.n))
        #     f.write('\t"ne": {},\n'.format(self.ne))
        #     f.write('\t"learning_rate": {},\n'.format(self.learning_rate))
        #     f.write('\t"learning_step": {},\n'.format(self.learning_step))
        #     f.write('\t"epoch": {},\n'.format(self.epoch))
        #     # f.write('\t"jax": {},\n'.format(str(self.jax)))
        #     f.write('\t"min_k": {},\n'.format(self.min_k))
        #     f.write('\t"max_k": {},\n'.format(self.max_k))
        #     f.write('\t"indices_source": {},\n'.format(self.indices_source.tolist()))
        #     f.write('\t"inputs_source": {},\n'.format(self.inputs_source.tolist()))
        #     f.write('\t"indices_target": {},\n'.format(self.indices_target.tolist()))
        #     f.write('\t"outputs_target": {},\n'.format(self.outputs_target.tolist()))
        #     f.write('\t"target_type": "{}",\n'.format(self.target_type))

        #     f.write('\t"losses": {},\n'.format(self.losses))
        #     f.write('\t"end_epoch": {}\n'.format(self.end_epoch))
        #     f.write('}')

    def save_local(self, path):
        ''' Save the current conductances in CSV format. '''
        # if the file already exists, append the conductances to the file
        # if os.path.isfile(path):
        #     with open(path, 'a') as f:
        #         f.write('{}\n'.format(self.conductances.tolist()))
        # # if the file does not exist, create it and write the conductances
        # else:
        #     with open(path, 'w') as f:
        #         f.write('{}\n'.format(self.conductances.tolist()))
        if self.learning_step == 0:
            save_to_csv(path, self.conductances.tolist(), mode='w')
        else:
            save_to_csv(path, self.conductances.tolist())

        



    '''
	*****************************************************************************************************
	*****************************************************************************************************

										PLOTTING AND ANIMATION

	*****************************************************************************************************
	*****************************************************************************************************
    '''

    def plot_circuit(self, title=None, lw = 0.5, point_size = 100, highlight_nodes = False, figsize = (4,4), highlighted_point_size = 200, filename = None):
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
            # sources in red
            axs.scatter(posX[self.indices_source], posY[self.indices_source], s = highlighted_point_size, c = 'red', zorder = 10)
            # targets in blue. Check the type of target
            if self.target_type == 'node':
                axs.scatter(posX[self.indices_target], posY[self.indices_target], s = highlighted_point_size, c = 'blue', zorder = 10)
            elif self.target_type == 'edge':
                axs.scatter(posX[self.indices_target[:,0]], posY[self.indices_target[:,0]], s = highlighted_point_size, c = 'blue', zorder = 10)
                axs.scatter(posX[self.indices_target[:,1]], posY[self.indices_target[:,1]], s = 0.5*highlighted_point_size, c = 'blue', zorder = 10)
            # try:
            #     axs.scatter(posX[self.indices_target[:,1:]], posY[self.indices_target[:,1:]], s = highlighted_point_size, c = 'blue', zorder = 10)
            # except:
            #     axs.scatter(posX[self.indices_target], posY[self.indices_target], s = highlighted_point_size, c = 'blue', zorder = 10)
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

def save_to_csv(filename, data, mode='a'):
    """
    Save data to a CSV file.
    
    Parameters:
    - filename: Name of the file to save to.
    - data: The data to save (should be a list or array).
    - mode: File mode ('w' for write, 'a' for append). Default is 'a'.
    """
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def load_from_csv(filename):
    """
    Load data from a CSV file.
    
    Parameters:
    - filename: Name of the file to load from.
    
    Returns:
    - A list of lists containing the data.
    """
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(item) for item in row])
    return data


def CL_from_file(jsonfile_global, jsonfile_graph, csv_local=None):
    # create a CL object from a json file
    with open(jsonfile_global, 'r') as f:
        data_global = json.load(f)
    with open(jsonfile_graph, 'r') as f:
        data_graph = json.load(f)
    if csv_local:
        conductances = load_from_csv(csv_local)[-1]
    
    # extract the attributes
    graph = network_from_json(jsonfile_graph)
    learning_rate = data_global['learning_rate']
    learning_step = data_global['learning_step']
    min_k = data_global['min_k']
    max_k = data_global['max_k']
    name = data_global['name']
    jax = data_global['jax']
    losses = data_global['losses']
    end_epoch = data_global['end_epoch']

    # extract the task
    indices_source = data_global['indices_source']
    inputs_source = data_global['inputs_source']
    indices_target = data_global['indices_target']
    outputs_target = data_global['outputs_target']
    target_type = data_global['target_type']

    if jax:
        conductances = jnp.array(conductances)
        indices_source = jnp.array(indices_source)
        inputs_source = jnp.array(inputs_source)
        indices_target = jnp.array(indices_target)
        outputs_target = jnp.array(outputs_target)

    allo = CL(graph, conductances, learning_rate, learning_step, min_k, max_k, name, jax, losses, end_epoch)
    if jax:
        allo.jax_set_task(indices_source, inputs_source, indices_target, outputs_target, target_type)
    else:
        allo.set_task(indices_source, inputs_source, indices_target, outputs_target, target_type)   

    return allo