from cl_utils import *
import jax
import json
import csv
from jax import jit

class restricted_CL(CL):
    @staticmethod
    @jit
    def _threshold_sstep_GD_NA(threshold,conductances, incidence_matrix, Q, inputs_source, indices_target, outputs_target, learning_rate, min_k, max_k):
        ''' Perform a step of gradient descent over the MSE for Node Allostery '''
        grad_vec = jax.grad(CL.MSE_NA)(conductances, incidence_matrix, Q, inputs_source, indices_target, outputs_target)
        mask = jnp.abs(grad_vec) > threshold
        grad_vec = grad_vec*mask
        new_conductances = conductances - learning_rate*grad_vec
        new_conductances = jnp.clip(new_conductances, min_k, max_k)
        return new_conductances

    @staticmethod
    @jit
    def _threshold_sstep_GD_EA(threshold,conductances, incidence_matrix, Q, inputs_source, indices_target, outputs_target, learning_rate, min_k, max_k):
        ''' Perform a step of gradient descent over the MSE for Edge Allostery '''
        grad_vec = jax.grad(CL.MSE_EA)(conductances, incidence_matrix, Q, inputs_source, indices_target, outputs_target)
        mask = jnp.abs(grad_vec) > threshold
        grad_vec = grad_vec*mask
        new_conductances = conductances - learning_rate*grad_vec
        new_conductances = jnp.clip(new_conductances, min_k, max_k)
        return new_conductances
    def _siterate_GD(self, threshold, n_steps, task):
        ''' Iterate gradient descent for n_steps.
        task is a string: "node" (node allostery) or "edge" (edge allostery)
        '''
        if task == "node":
            for i in range(n_steps):
                free_state = Circuit.ssolve(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source)
                voltage_drop_free = free_state.dot(self.incidence_matrix)
                self.current_power = np.sum(self.conductances*(voltage_drop_free**2))
                self.current_energy += self.current_power
                self.learning_step += 1
                self.conductances = restricted_CL._threshold_sstep_GD_NA(threshold,self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source, self.indices_target, self.outputs_target, self.learning_rate, self.min_k, self.max_k)
        elif task == "edge":
            for i in range(n_steps):
                free_state = Circuit.ssolve(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source)
                voltage_drop_free = free_state.dot(self.incidence_matrix)
                self.current_power = np.sum(self.conductances*(voltage_drop_free**2))
                self.current_energy += self.current_power
                self.learning_step += 1
                self.conductances = restricted_CL._threshold_sstep_GD_EA(threshold,self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source, self.indices_target, self.outputs_target, self.learning_rate, self.min_k, self.max_k)
        else:
            raise Exception('task must be "node" or "edge"')
        return self.conductances

    def train_GD(self, threshold, n_epochs, n_steps_per_epoch, verbose=True, pbar=False, log_spaced=False, save_global=False, save_state=False, save_path='trained_circuit'):
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
            loss = CL.MSE(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source, self.indices_target, self.outputs_target, self.target_type)
            if loss < self.best_error:
                self.best_error = loss
                self.best_conductances = self.conductances
            self.losses.append(loss)
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
            conductances = self._siterate_GD(threshold,actual_steps_per_epoch, self.target_type)
            loss = CL.MSE(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source, self.indices_target, self.outputs_target, self.target_type)
            if loss < self.best_error:
                self.best_error = loss
                self.best_conductances = self.conductances
            self.losses.append(loss)
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
        free_state = Circuit.ssolve(self.conductances, self.incidence_matrix, self.Q_free, self.inputs_source)
        voltage_drop_free = free_state.dot(self.incidence_matrix)
        self.current_power = np.sum(self.conductances*(voltage_drop_free**2))
        self.current_energy += self.current_power
        self.power.append(self.current_power)
        self.energy.append(self.current_energy)

        if save_global:
            self.save_global(save_path+'_global.json')
            self.save_graph(save_path+'_graph.json')
        return self.losses, conductances
