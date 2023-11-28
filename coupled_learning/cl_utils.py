import numpy as np
# from .circuit_utils import Circuit
from circuit_utils import Circuit
from network_utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from scipy.sparse import csr_matrix, csc_array
import jax.numpy as jnp
import jax
import json
import csv
from jax import jit

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
    def __init__(self, graph, conductances, learning_rate=1.0, learning_step = 0, min_k = 1.e-6, max_k = 1.e6, name = 'CL', jax = False, losses = None, end_epoch = None, power = None, energy = None):
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
        if end_epoch is None or end_epoch == []:
            self.end_epoch = []
            self.epoch = 0
        else:
            self.epoch = end_epoch[-1]
            self.end_epoch = end_epoch
        if power is None or power == []:
            self.power = []
            self.current_power = 0
        else:
            self.power = power
            self.current_power = power[-1]
        if energy is None or energy == []:
            self.energy = []
            self.current_energy = 0
        else:
            self.energy = energy
            self.current_energy = energy[-1]
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

    '''
	*****************************************************************************************************
	*****************************************************************************************************

										TASK

	*****************************************************************************************************
	*****************************************************************************************************
    '''

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
        assert indices_source.shape[0] == inputs_source.shape[-1], 'indices_source and each input set in inputs_source must have the same length'
        assert inputs_source.ndim == 1 or inputs_source.ndim == 2, 'inputs_source must be array of 1 or 2 dimensions'
        
        self.indices_source = indices_source
        if inputs_source.ndim==1:
            self.inputs_source = inputs_source.reshape(1,inputs_source.size)
        else:
            self.inputs_source = inputs_source

        self.ntasks = self.inputs_source.shape[0]

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
        assert indices_target.shape[0] == outputs_target.shape[-1], 'indices_target and outputs_target must have the same length'
        assert outputs_target.ndim == 1 or outputs_target.ndim == 2, 'inputs_source must be array of 1 or 2 dimensions'

        self.indices_target = indices_target
        if outputs_target.ndim==1:
            self.outputs_target = outputs_target.reshape(1,outputs_target.size)
        else:
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

        if self.jax:
            return self.jax_set_task(indices_source, inputs_source, indices_target, outputs_target, target_type)
        else:
            # Compute the constraint matrices
            self.Q_free = self.constraint_matrix(self.indices_source)
            if self.target_type == 'node':
                self.Q_clamped = self.constraint_matrix(np.concatenate((self.indices_source, self.indices_target)))    
            elif self.target_type == 'edge':
                q_edge = self.constraint_matrix(indices_target, restrictionType='edge')
                self.Q_clamped = hstack([self.Q_free, q_edge])

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
            q_edge = self.jax_constraint_matrix(indices_target, restrictionType='edge')
            self.Q_clamped = jnp.concatenate([self.Q_free,q_edge], axis=1)

        return self.Q_free, self.Q_clamped
    
    def get_free_state(self, input_idx=0):
        ''' Return the free state of the circuit for the first task. '''
        # determine if a task was given
        if not hasattr(self, 'Q_free'):
            raise ValueError('No task was given to the circuit. Use set_task(indices_source, inputs_source, indices_target, outputs_target) to set the task.')
        return self.solve(self.Q_free, self.inputs_source[input_idx])

    def get_power_state(self):
        ''' Return the power state of the circuit for the first task. '''
        free_state = self.get_free_state()
        voltage_drop_free = self.incidence_matrix.T.dot(free_state)
        return self.conductances*(voltage_drop_free**2)/2

    def MSE_loss_input(self, input_idx=0):
        ''' Compute the MSE loss for one input. '''

        free_state = self.get_free_state(input_idx)

        if self.target_type == 'node':
            return 0.5*np.mean((free_state[self.indices_target] - self.outputs_target[input_idx])**2)
        elif self.target_type == 'edge':
            # freeState_DV = free_state[self.indices_target[:,1]] - free_state[self.indices_target[:,2]]
            freeState_DV = free_state[self.indices_target[:,0]] - free_state[self.indices_target[:,1]]
            return 0.5*np.mean((freeState_DV - self.outputs_target[input_idx])**2)

    def MSE_loss(self):
        ''' Compute the MSE loss for all inputs. '''
        return np.sum([self.MSE_loss_input(input_idx) for input_idx in range(self.ntasks)])


    def jax_MSE_loss_input(self, conductances, input_idx):
        ''' Compute the MSE loss. '''
        print("This function is deprecated. Use MSE instead.")
        self.setConductances(conductances)
        free_state = self.jax_solve(self.Q_free, self.inputs_source[input_idx])
        if self.target_type == 'node':
            return 0.5*jnp.mean((free_state[self.indices_target] - self.outputs_target[input_idx])**2)
        elif self.target_type == 'edge':
            # freeState_DV = free_state[self.indices_target[:,1]] - free_state[self.indices_target[:,2]]
            freeState_DV = free_state[self.indices_target[:,0]] - free_state[self.indices_target[:,1]]
            return 0.5*jnp.mean((freeState_DV - self.outputs_target)**2)

    def jax_MSE_loss(self, conductances):
        ''' Compute the MSE loss for all inputs. '''
        return jnp.sum([self.jax_MSE_loss_input(conductances, input_idx) for input_idx in range(self.ntasks)])


    # static methods for jit compilation
    @staticmethod
    @jit
    def MSE_NA(conductances, incidence_matrix, Q, inputs_source, indices_target, outputs_target):
        ''' Compute the MSE loss for Node Allostery. '''
        free_state = Circuit.ssolve(conductances, incidence_matrix, Q, inputs_source)
        return 0.5*jnp.mean((free_state[indices_target] - outputs_target)**2)

    @staticmethod
    @jit
    def MSE_EA(conductances, incidence_matrix, Q, inputs_source, indices_target, outputs_target):
        ''' Compute the MSE loss for Edge Allostery'''
        free_state = Circuit.ssolve(conductances, incidence_matrix, Q, inputs_source)
        freeState_DV = free_state[indices_target[:,0]] - free_state[indices_target[:,1]]
        return 0.5*jnp.mean((freeState_DV - outputs_target)**2)
    
    @staticmethod
    def MSE(conductances, incidence_matrix, Q, inputs_source, indices_target, outputs_target, target_type):
        ''' Compute the MSE loss for Node Allostery or Edge Allostery'''
        if target_type == 'node':
            return jnp.array([CL.MSE_NA(conductances, incidence_matrix, Q, inputs_source[input_idx], indices_target, outputs_target[input_idx]) for input_idx in range(inputs_source.shape[0])]).sum()
        elif target_type == 'edge':
            return jnp.array([CL.MSE_EA(conductances, incidence_matrix, Q, inputs_source[input_idx], indices_target, outputs_target[input_idx]) for input_idx in range(inputs_source.shape[0])]).sum()
        else:
            raise Exception('target_type must be "node" or "edge"' )

    @staticmethod
    def gradient_MSE(conductances, incidence_matrix, Q, inputs_source, indices_target, outputs_target, target_type):
        ''' Compute the gradient of the MSE loss for Node Allostery or Edge Allostery'''
        if target_type == 'node' or target_type == 'edge':
            grad_func = jax.grad(CL.MSE, argnums=0)
        else:
            raise ValueError('target_type must be "node" or "edge"')
        return grad_func(conductances, incidence_matrix, Q, inputs_source, indices_target, outputs_target, target_type)

    @staticmethod
    def hessian_MSE(conductances, incidence_matrix, Q, inputs_source, indices_target, outputs_target, target_type):
        ''' Compute the hessian of the MSE loss for Node Allostery or Edge Allostery'''
        if target_type == 'node' or target_type == 'edge':
            hessian_func = jax.hessian(CL.MSE, argnums=0)
        else:
            raise ValueError('target_type must be "node" or "edge"')
        return hessian_func(conductances, incidence_matrix, Q, inputs_source, indices_target, outputs_target, target_type)


    def jaxify(self):
        ''' Jaxify the circuit. '''
        
        converted = False
        if not self.jax:
            self.jax = True
            converted = True
        
        if not isinstance(self.Q_free, jnp.ndarray):
            self.Q_free = jnp.array(self.Q_free.todense())
            converted = True

        if not isinstance(self.Q_clamped, jnp.ndarray):
            self.Q_clamped = jnp.array(self.Q_clamped.todense())
            converted = True

        if not isinstance(self.incidence_matrix, jnp.ndarray):
            self.incidence_matrix = jnp.array(self.incidence_matrix.todense())
            converted = True

        if not isinstance(self.conductances, jnp.ndarray):
            self.conductances = jnp.array(self.conductances)
            converted = True

        if converted:
            print('Converted to jax')
        else:
            print('Already jaxified')

    def sparsify(self):
        ''' Sparsify the circuit. '''
        converted = False
        if self.jax:
            self.jax = False
            converted = True
        
        if isinstance(self.Q_free, jnp.ndarray):
            self.Q_free = csr_matrix(self.Q_free, dtype = np.float64)
            converted = True

        if isinstance(self.Q_clamped, jnp.ndarray):
            self.Q_clamped = csr_matrix(self.Q_clamped, dtype = np.float64)
            converted = True

        if isinstance(self.incidence_matrix, jnp.ndarray):
            self.incidence_matrix = csc_array(self.incidence_matrix, dtype = np.float64)
            converted = True

        if isinstance(self.conductances, jnp.ndarray):
            self.conductances = np.array(self.conductances, dtype = np.float64)
            converted = True

        if converted:
            print('Converted to sparse')
        else:
            print('Already sparse')
        
        
    
    '''
	*****************************************************************************************************
	*****************************************************************************************************

								TRAINING: COUPLED LEARNING AND GRADIENT DESCENT

	*****************************************************************************************************
	*****************************************************************************************************
    '''


    ########################################        
    ########### COUPLED LEARNING ############
    ########################################




    def _step_CL(self, eta = 0.001, batchSize=1):
        ''' Perform a step of coupled learning. '''
        # free state. Notice that it is not worth using get_free_state, since it checks if a task was given at each call.
        
        transpose_incidence_matrix = self.incidence_matrix.T
        voltage_drop_free = np.zeros(self.ne)
        voltage_drop_clamped = np.zeros(self.ne)
        delta_conductances = np.zeros(self.ne)
        averageInferencePower = 0

        batchInputs = np.random.choice(self.ntasks, batchSize, replace=False)
        for batchInput in batchInputs:
            free_state = self.solve(self.Q_free, self.inputs_source[batchInput])

            if self.target_type == 'node':
                nudge = free_state[self.indices_target] + eta * (self.outputs_target[batchInput] - free_state[self.indices_target])

            elif self.target_type == 'edge':

                # DP = free_state[self.indices_target[:,1]] - free_state[self.indices_target[:,2]]
                DP = free_state[self.indices_target[:,0]] - free_state[self.indices_target[:,1]]
                nudge = DP + eta * (self.outputs_target[batchInput] - DP)

            clamped_state = self.solve(self.Q_clamped, np.concatenate((self.inputs_source[batchInput], nudge)))

            # voltage drop
            # ROOM FOR IMPROVEMENT? WE ARE TRANSPOSING THE INCIDENCE MATRIX AT EACH STEP
            voltage_drop_free = transpose_incidence_matrix.dot(free_state)/batchSize
            voltage_drop_clamped = transpose_incidence_matrix.dot(clamped_state)/batchSize
            
            # Update the conductances
            delta_conductances += -1.0/eta * (voltage_drop_clamped**2 - voltage_drop_free**2)

            averageInferencePower += np.sum(self.conductances*(voltage_drop_free**2)/2)/batchSize
        # power
        self.current_power += averageInferencePower
        # energy
        self.current_energy += self.current_power

        self.conductances = self.conductances + self.learning_rate*delta_conductances
        self._clip_conductances()
        # Update the learning step
        self.learning_step += 1
        
        return free_state, voltage_drop_free, delta_conductances, self.conductances

    def iterate_CL(self, n_steps, eta = 0.001, batchSize=1):
        ''' Iterate coupled learning for n_steps. '''
        for i in range(n_steps):
            free_state, voltage_drop_free , delta_conductances , conductances = self._step_CL(eta, batchSize)
        return free_state, voltage_drop_free , delta_conductances , conductances
    

    def train(self, n_epochs, n_steps_per_epoch, eta = 0.001, batchSize = 1, verbose = True, pbar = False, log_spaced = False, save_global = False, save_state = False, save_path = 'trained_circuit'):
        ''' Train the circuit for n_epochs. Each epoch consists of n_steps_per_epoch steps of coupled learning.
        If log_spaced is True, n_steps_per_epoch is overwritten and the number of steps per epoch is log-spaced, such that the total number of steps is n_steps_per_epoch * n_epochs.
        '''
        if pbar:
            epochs = tqdm(range(n_epochs))
        else:
            epochs = range(n_epochs)

        # initial state
        if self.learning_step == 0:
            self.end_epoch.append(self.learning_step)
            self.losses.append(self.MSE_loss())
            if save_state:
                self.save_local(save_path+'.csv')
        else: #to avoid double counting the initial state
            # remove the last element of power and energy
            self.power.pop()
            self.energy.pop()
            # set the current power and energy to the last element
            self.current_power = self.power[-1]
            self.current_energy = self.energy[-1]

        if log_spaced:
            n_steps = n_epochs * n_steps_per_epoch
            n_steps_per_epoch = log_partition(n_steps, n_epochs)
        else:
            actual_steps_per_epoch = n_steps_per_epoch
        
        #training
        for epoch in epochs:
            if log_spaced:
                actual_steps_per_epoch = n_steps_per_epoch[epoch]
            free_state, voltage_drop_free , delta_conductances , conductances = self.iterate_CL(actual_steps_per_epoch, eta, batchSize)
            self.losses.append(self.MSE_loss())
            self.power.append(self.current_power)
            self.energy.append(self.current_energy)
            if verbose:
                print('Epoch: {}/{} | Loss: {}'.format(epoch,n_epochs-1, self.losses[-1]))
            self.epoch += 1
            self.end_epoch.append(self.learning_step)
            if save_state:
                # self.save(save_path+'_epoch_'+str(epoch)+'.pkl')
                self.save_local(save_path+'.csv')
    
        # at the end of training, compute the current power and current energy, and save global and save graph
        self.current_power = np.sum(self.conductances*(voltage_drop_free**2)/2)
        self.current_energy += self.current_power
        self.power.append(self.current_power)
        self.energy.append(self.current_energy)

        if save_global:
            self.save_global(save_path+'_global.json')
            self.save_graph(save_path+'_graph.json')
        return self.losses, free_state, voltage_drop_free , delta_conductances , conductances

    @staticmethod
    @jit
    def _sstep_CL_NA(conductances, incidence_matrix, Q_free, Q_clamped, inputs_source, indices_target, outputs_target, learning_rate, min_k, max_k, eta):
        ''' Perform a step of coupled learning for Node Allostery. '''
        free_state = Circuit.ssolve(conductances, incidence_matrix, Q_free, inputs_source[0])

        nudge = free_state[indices_target] + eta * (outputs_target - free_state[indices_target])

        clamped_state = Circuit.ssolve(conductances, incidence_matrix,Q_clamped, jnp.concatenate((inputs_source, nudge)))

        # ROOM FOR IMPROVEMENT? WE ARE TRANSPOSING THE INCIDENCE MATRIX AT EACH STEP
        voltage_drop_free = incidence_matrix.T.dot(free_state)
        voltage_drop_clamped = incidence_matrix.T.dot(clamped_state)

        # power
        current_power = np.sum(conductances*(voltage_drop_free**2)/2)

        # Update the conductances
        delta_conductances = -1.0/eta * (voltage_drop_clamped**2 - voltage_drop_free**2)
        new_conductances = conductances + learning_rate*delta_conductances
        new_conductances = jnp.clip(new_conductances, min_k, max_k)

        return free_state, voltage_drop_free, delta_conductances, new_conductances, current_power

    @staticmethod
    @jit
    def _sstep_CL_EA(conductances, incidence_matrix, Q_free, Q_clamped, inputs_source, indices_target, outputs_target, learning_rate, min_k, max_k, eta):
        ''' Perform a step of coupled learning for Edge Allostery '''
        free_state = Circuit.ssolve(conductances, incidence_matrix, Q_free, inputs_source[0])

        DP = free_state[indices_target[:,0]] - free_state[indices_target[:,1]]
        nudge = DP + eta * (outputs_target - DP)

        clamped_state = Circuit.ssolve(conductances, incidence_matrix,Q_clamped, jnp.concatenate((inputs_source, nudge)))

        # ROOM FOR IMPROVEMENT? WE ARE TRANSPOSING THE INCIDENCE MATRIX AT EACH STEP
        voltage_drop_free = incidence_matrix.T.dot(free_state)
        voltage_drop_clamped = incidence_matrix.T.dot(clamped_state)

        # power
        current_power = np.sum(conductances*(voltage_drop_free**2)/2)

        # Update the conductances
        delta_conductances = -1.0/eta * (voltage_drop_clamped**2 - voltage_drop_free**2)
        new_conductances = conductances + learning_rate*delta_conductances
        new_conductances = jnp.clip(new_conductances, min_k, max_k)

        return free_state, voltage_drop_free, delta_conductances, new_conductances, current_power

    def _siterate_CL(self, n_steps, eta, task):
        ''' Iterate coupled learning for n_steps.
        task is a string: "node" (node allostery) or "edge" (edge allostery)
        '''
        if task == "node":
            for i in range(n_steps):
                _, _, _, self.conductances, current_power = CL._sstep_CL_NA(self.conductances, self.incidence_matrix, self.Q_free, self.Q_clamped, self.inputs_source, self.indices_target, self.outputs_target, self.learning_rate, self.min_k, self.max_k, eta)
                self.current_power = current_power
                self.current_energy += self.current_power
                self.learning_step += 1
        elif task == "edge":
            for i in range(n_steps):
                _, _, _, self.conductances, current_power = CL._sstep_CL_EA(self.conductances, self.incidence_matrix, self.Q_free, self.Q_clamped, self.inputs_source, self.indices_target, self.outputs_target, self.learning_rate, self.min_k, self.max_k, eta)
                self.current_power = current_power
                self.current_energy += self.current_power
                self.learning_step += 1
        else:
            raise Exception('task must be "node" or "edge"')
        return self.conductances

    
    def train_CL(self, n_epochs, n_steps_per_epoch, eta, verbose = True, pbar = False, log_spaced = False, save_global = False, save_state = False, save_path = 'trained_circuit'):
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
        else:
            actual_steps_per_epoch = n_steps_per_epoch

        if self.jax: # Dense and JIT training (gpu)
            # initial state
            if self.learning_step == 0:
                self.end_epoch.append(self.learning_step)
                self.losses.append(CL.MSE(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source, self.indices_target, self.outputs_target, self.target_type))
                if save_state:
                    self.save_local(save_path+'.csv')
            else: #to avoid double counting the initial state
                # remove the last element of power and energy
                self.power.pop()
                self.energy.pop()
                # set the current power and energy to the last element
                self.current_power = self.power[-1]
                self.current_energy = self.energy[-1]

            #training
            for epoch in epochs:
                if log_spaced:
                    actual_steps_per_epoch = n_steps_per_epoch[epoch]
                conductances = self._siterate_CL(actual_steps_per_epoch, eta, self.target_type)
                self.losses.append(CL.MSE(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source, self.indices_target, self.outputs_target, self.target_type))
                self.power.append(self.current_power)
                self.energy.append(self.current_energy)
                if verbose:
                    print('Epoch: {}/{} | Loss: {}'.format(epoch,n_epochs-1, self.losses[-1]))
                self.epoch += 1
                self.end_epoch.append(self.learning_step)
                if save_state:
                    # self.save(save_path+'_epoch_'+str(epoch)+'.pkl')
                    self.save_local(save_path+'.csv')

            # at the end of training, compute the current power and current energy, and save global and save graph
            free_state = Circuit.ssolve(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source[0])
            voltage_drop_free = free_state.dot(self.incidence_matrix)
            self.current_power = np.sum(self.conductances*(voltage_drop_free**2)/2)
            self.current_energy += self.current_power
            self.power.append(self.current_power)
            self.energy.append(self.current_energy)

            if save_global:
                self.save_global(save_path+'_global.json')
                self.save_graph(save_path+'_graph.json')
            return self.losses, conductances
        else: # Sparse and no JIT training (no gpu)
            # initial state
            if self.learning_step == 0:
                self.end_epoch.append(self.learning_step)
                self.losses.append(self.MSE_loss())
                if save_state:
                    self.save_local(save_path+'.csv')
            else: #to avoid double counting the initial state
                # remove the last element of power and energy
                self.power.pop()
                self.energy.pop()
                # set the current power and energy to the last element
                self.current_power = self.power[-1]
                self.current_energy = self.energy[-1]

            #training
            for epoch in epochs:
                if log_spaced:
                    actual_steps_per_epoch = n_steps_per_epoch[epoch]
                free_state, voltage_drop_free , delta_conductances , conductances = self.iterate_CL(actual_steps_per_epoch, eta)
                self.losses.append(self.MSE_loss())
                self.power.append(self.current_power)
                self.energy.append(self.current_energy)
                if verbose:
                    print('Epoch: {}/{} | Loss: {}'.format(epoch,n_epochs-1, self.losses[-1]))
                self.epoch += 1
                self.end_epoch.append(self.learning_step)
                if save_state:
                    # self.save(save_path+'_epoch_'+str(epoch)+'.pkl')
                    self.save_local(save_path+'.csv')

            # at the end of training, compute the current power and current energy, and save global and save graph
            self.current_power = np.sum(self.conductances*(voltage_drop_free**2)/2)
            self.current_energy += self.current_power
            self.power.append(self.current_power)
            self.energy.append(self.current_energy)

            if save_global:
                self.save_global(save_path+'_global.json')
                self.save_graph(save_path+'_graph.json')
            return self.losses, conductances






    ########################################        
    ########### GRADIENT DESCENT ############
    ########################################





    @staticmethod
    @jit
    def _sstep_GD_NA(conductances, incidence_matrix, Q, inputs_source, indices_target, outputs_target, batchInputIdx, learning_rate, min_k, max_k):
        ''' Perform a step of gradient descent over the MSE for Node Allostery '''
        batchGradientDirection = jnp.array([jax.grad(CL.MSE_NA)(conductances, incidence_matrix, Q, inputs_source[inputIdx], indices_target, outputs_target[inputIdx]) for inputIdx in batchInputIdx])
        new_conductances = conductances - learning_rate*jnp.mean(batchGradientDirection)
        new_conductances = jnp.clip(new_conductances, min_k, max_k)
        return new_conductances

    @staticmethod
    @jit
    def _sstep_GD_EA(conductances, incidence_matrix, Q, inputs_source, indices_target, outputs_target, batchInputIdx, learning_rate, min_k, max_k):
        ''' Perform a step of gradient descent over the MSE for Edge Allostery '''       
        batchGradientDirection = jnp.array([jax.grad(CL.MSE_EA)(conductances, incidence_matrix, Q, inputs_source[inputIdx], indices_target, outputs_target[inputIdx]) for inputIdx in batchInputIdx])
        new_conductances = conductances - learning_rate*jnp.mean(batchGradientDirection)
        new_conductances = jnp.clip(new_conductances, min_k, max_k)
        return new_conductances

    def _siterate_GD(self, n_steps, task, batchSize=1):
        ''' Iterate gradient descent for n_steps.
        task is a string: "node" (node allostery) or "edge" (edge allostery)
        '''
        if task == "node":
            for i in range(n_steps):
                free_state = Circuit.ssolve(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source[0])
                voltage_drop_free = free_state.dot(self.incidence_matrix)
                self.current_power = np.sum(self.conductances*(voltage_drop_free**2)/2)
                self.current_energy += self.current_power
                self.learning_step += 1
                batchInputIdx = np.random.choice(self.inputs_source.shape[0], batchSize, replace=False)
                self.conductances = CL._sstep_GD_NA(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source, self.indices_target, self.outputs_target, batchInputIdx, self.learning_rate, self.min_k, self.max_k)
        elif task == "edge":
            for i in range(n_steps):
                free_state = Circuit.ssolve(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source[0])
                voltage_drop_free = free_state.dot(self.incidence_matrix)
                self.current_power = np.sum(self.conductances*(voltage_drop_free**2)/2)
                self.current_energy += self.current_power
                self.learning_step += 1
                batchInputIdx = np.random.choice(self.inputs_source.shape[0], batchSize, replace=False)
                self.conductances = CL._sstep_GD_EA(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source, self.indices_target, self.outputs_target, batchInputIdx, self.learning_rate, self.min_k, self.max_k)
        else:
            raise Exception('task must be "node" or "edge"')
        return self.conductances


    def train_GD(self, n_epochs, n_steps_per_epoch, batchSize=1, verbose = True, pbar = False, log_spaced = False, save_global = False, save_state = False, save_path = 'trained_circuit'):
        ''' Train the circuit for n_epochs. Each epoch consists of n_steps_per_epoch steps of gradient descent.
        If log_spaced is True, n_steps_per_epoch is overwritten and the number of steps per epoch is log-spaced, such that the total number of steps is n_steps_per_epoch * n_epochs.
        '''
        if pbar:
            epochs = tqdm(range(n_epochs))
        else:
            epochs = range(n_epochs)

        if log_spaced:
            n_steps = n_epochs * n_steps_per_epoch
            n_steps_per_epoch = log_partition(n_steps, n_epochs)
        else:
            actual_steps_per_epoch = n_steps_per_epoch
        


        # initial state
        if self.learning_step == 0:
            self.end_epoch.append(self.learning_step)
            self.losses.append(CL.MSE(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source, self.indices_target, self.outputs_target, self.target_type))
            if save_state:
                self.save_local(save_path+'.csv')
        else: #to avoid double counting the initial state
            # remove the last element of power and energy
            self.power.pop()
            self.energy.pop()
            # set the current power and energy to the last element
            self.current_power = self.power[-1]
            self.current_energy = self.energy[-1]

        #training
        for epoch in epochs:
            if log_spaced:
                actual_steps_per_epoch = n_steps_per_epoch[epoch]
            conductances = self._siterate_GD(actual_steps_per_epoch, self.target_type, batchSize)
            self.losses.append(CL.MSE(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source, self.indices_target, self.outputs_target, self.target_type))
            self.power.append(self.current_power)
            self.energy.append(self.current_energy)
            if verbose:
                print('Epoch: {}/{} | Loss: {}'.format(epoch,n_epochs-1, self.losses[-1]))
            self.epoch += 1
            self.end_epoch.append(self.learning_step)
            if save_state:
                # self.save(save_path+'_epoch_'+str(epoch)+'.pkl')
                self.save_local(save_path+'.csv')
        # at the end of training, compute the current power and current energy, and save global and save graph
        free_state = Circuit.ssolve(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source[0])
        voltage_drop_free = free_state.dot(self.incidence_matrix)
        self.current_power = np.sum(self.conductances*(voltage_drop_free**2)/2)
        self.current_energy += self.current_power
        self.power.append(self.current_power)
        self.energy.append(self.current_energy)

        if save_global:
            self.save_global(save_path+'_global.json')
            self.save_graph(save_path+'_graph.json')
        return self.losses, conductances


    def reset_training(self):
        ''' Reset the training. '''
        self.learning_step = 0
        self.epoch = 0
        self.end_epoch = []
        self.losses = []
        self.power = []
        self.energy = []
        self.current_power = 0
        self.current_energy = 0


    # JUST FOR THE RECORD. TO BE DELETED AT SOME POINT

    # def _step_GD(self):
    #     ''' Perform a step of gradient descent over the MSE. '''
    #     # free state. Notice that it is not worth using get_free_state, since it checks if a task was given at each call.
    #     free_state = self.solve(self.Q_free, self.inputs_source)
    #     # power and energy prior to the update
    #     # WARNING: I'M NOT SURE WHY JAX DOT OF (N,) AND (M, N) GIVES THE SAME RESULT AS NUMPY DOT  (M, N) AND (N,), BUT IT SAVES A TRANSPOSITION
    #     voltage_drop_free = free_state.dot(self.incidence_matrix)
    #     self.current_power = np.sum(self.conductances*(voltage_drop_free**2)/2)
    #     self.current_energy += self.current_power
    #     # update the conductances
    #     self.conductances = self.conductances - self.learning_rate*jax.grad(self.jax_MSE_loss)(self.conductances)
    #     self.conductances = jnp.clip(self.conductances, self.min_k, self.max_k)
    #     self.learning_step += 1

    #     return self.conductances
        

    # def iterate_GD(self, n_steps):
    #     ''' Iterate gradient descent for n_steps. '''
    #     for i in range(n_steps):
    #         conductances = self._step_GD()
    #     return conductances

    # def train_GD(self, n_epochs, n_steps_per_epoch, verbose = True, pbar = False, log_spaced = False, save_global = False, save_state = False, save_path = None):
    #     ''' Train the circuit for n_epochs. Each epoch consists of n_steps_per_epoch steps of gradient descent.
    #     If log_spaced is True, n_steps_per_epoch is overwritten and the number of steps per epoch is log-spaced, such that the total number of steps is n_steps_per_epoch * n_epochs.
    #     '''
    #     if pbar:
    #         epochs = tqdm(range(n_epochs))
    #     else:
    #         epochs = range(n_epochs)

    #     if log_spaced:
    #         n_steps = n_epochs * n_steps_per_epoch
    #         n_steps_per_epoch = log_partition(n_steps, n_epochs)
    #     else:
    #         actual_steps_per_epoch = n_steps_per_epoch
        


    #     # initial state
    #     if self.learning_step == 0:
    #         self.end_epoch.append(self.learning_step)
    #         self.losses.append(float(self.jax_MSE_loss(self.conductances)))
    #         if save_state:
    #             self.save_local(save_path+'.csv')
    #     else: #to avoid double counting the initial state
    #         # remove the last element of power and energy
    #         self.power.pop()
    #         self.energy.pop()
    #         # set the current power and energy to the last element
    #         self.current_power = self.power[-1]
    #         self.current_energy = self.energy[-1]

    #     #training
    #     for epoch in epochs:
    #         if log_spaced:
    #             actual_steps_per_epoch = n_steps_per_epoch[epoch]
    #         conductances = self.iterate_GD(actual_steps_per_epoch)
    #         self.losses.append(float(self.jax_MSE_loss(conductances)))
    #         self.power.append(self.current_power)
    #         self.energy.append(self.current_energy)
    #         if verbose:
    #             print('Epoch: {}/{} | Loss: {}'.format(epoch,n_epochs-1, self.losses[-1]))
    #         self.epoch += 1
    #         self.end_epoch.append(self.learning_step)
    #         if save_state:
    #             # self.save(save_path+'_epoch_'+str(epoch)+'.pkl')
    #             self.save_local(save_path+'.csv')
    #     # at the end of training, compute the current power and current energy, and save global and save graph
    #     voltage_drop_free = self.get_free_state().dot(self.incidence_matrix)
    #     self.current_power = np.sum(self.conductances*(voltage_drop_free**2)/2)
    #     self.current_energy += self.current_power
    #     self.power.append(self.current_power)
    #     self.energy.append(self.current_energy)

    #     if save_global:
    #         self.save_global(save_path+'_global.json')
    #         self.save_graph(save_path+'_graph.json')
    #     return self.losses, conductances

    '''
	*****************************************************************************************************
	*****************************************************************************************************

										PRUNE AND REWIRE

	*****************************************************************************************************
	*****************************************************************************************************
    '''

    def prune_edge(self, edge):
        ''' Prune an edge of the circuit. '''
        self._remove_edge(edge)
        # reset the incidence matrix
        self.incidence_matrix = nx.incidence_matrix(self.graph, oriented=True)
        if jax:
            self.incidence_matrix = jnp.array(self.incidence_matrix.todense())

        # reset the task
        if jax:
            self.jax_set_task(self.indices_source, self.inputs_source, self.indices_target, self.outputs_target, self.target_type)
        else:
            self.set_task(self.indices_source, self.inputs_source, self.indices_target, self.outputs_target, self.target_type)

    def prune_edge_bunch(self, edge):
        ''' Prune an edge of the circuit. '''
        pass
    
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

    def save_global(self, path):
        ''' Save the attributes of the circuit in JSON format. '''
        # create a dictionary with the attributes
        if jax:
            losses = jax.device_get(jnp.array(self.losses)).astype(float).tolist()
            energies = jax.device_get(jnp.array(self.energy)).astype(float).tolist()
            powers = jax.device_get(jnp.array(self.power)).astype(float).tolist()
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
            "losses": losses,
            "energy": energies,
            "power": powers,
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




'''
*****************************************************************************************************
*****************************************************************************************************

									UTILS

*****************************************************************************************************
*****************************************************************************************************
'''


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


def CL_from_file(jsonfile_global, jsonfile_graph, csv_local=None, new_train=False):
    # create a CL object from a json file
    with open(jsonfile_global, 'r') as f:
        data_global = json.load(f)
    with open(jsonfile_graph, 'r') as f:
        data_graph = json.load(f)
    
    # extract the attributes
    graph = network_from_json(jsonfile_graph)
    if csv_local:
        conductances = load_from_csv(csv_local)[-1]
    else:
        conductances = np.ones(len(graph.edges))

    learning_rate = data_global['learning_rate']
    min_k = data_global['min_k']
    max_k = data_global['max_k']
    name = data_global['name']
    jax = data_global['jax']

    if new_train:
        losses = None
        energies = None
        powers = None
        end_epoch = None
        learning_step = 0
    else:
        losses = data_global['losses']
        energies = data_global['energy']
        powers = data_global['power']
        end_epoch = data_global['end_epoch']
        learning_step = data_global['learning_step']

    # extract the task
    indices_source = np.array(data_global['indices_source'])
    inputs_source = np.array(data_global['inputs_source'])
    indices_target = np.array(data_global['indices_target'])
    outputs_target = np.array(data_global['outputs_target'])
    target_type = data_global['target_type']

    if jax:
        conductances = jnp.array(conductances)
        indices_source = jnp.array(indices_source)
        inputs_source = jnp.array(inputs_source)
        indices_target = jnp.array(indices_target)
        outputs_target = jnp.array(outputs_target)

    allo = CL(graph, conductances, learning_rate, learning_step, min_k, max_k, name, jax, losses, end_epoch, power = powers, energy = energies)
    if jax:
        allo.jax_set_task(indices_source, inputs_source, indices_target, outputs_target, target_type)
    else:
        allo.set_task(indices_source, inputs_source, indices_target, outputs_target, target_type)   

    return allo