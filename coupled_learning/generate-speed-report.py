import sys
sys.path.append('../coupled_learning/')
from circuit_utils import *
from network_utils import *
from cl_utils import *
import matplotlib.pyplot as plt
from packing_utils import *
import time

import jax

def generate_speed_report():
    """
    Generates a speed report for the Coupled Learning (CL) and Gradient Descent (GD) algorithms on a grid network of different sizes.
    The report includes the size of the grid network, the number of nodes and edges, and the training time for both algorithms.
    The results are saved in a txt file named 'speed_results.txt' in the same directory as this script.

    Returns:
    None
    """
    # Check if JAX is using the CPU or GPU
    device = jax.devices()[0].platform
    if device == 'cpu':
        device_type = 'CPU'
    else:
        device_type = 'GPU'


    # Create the header of the table
    table_header = "Size (rows x cols) | Number of nodes | Number of edges | Training time for CL sparse without JIT (s) | Training time for CL dense with JIT (s) | Training time for GD dense with JIT (s)\n"
    table_header += "---------------------------------------------------------------------------------------------\n"

    # Create the file or overwrite it if it already exists
    with open('speed_results.txt', 'w') as f:
        f.write(f"Device used: {device_type}\n")
        f.write(table_header)

    size_array = np.array([[10,10],[20,20],[30,30],[40,40]])

    for size in size_array:
        g = grid_network(size[0], size[1], periodic=False, size_uc = (1,1), relabel_nodes=True)
        lr = 0.1
        initial_conductances = np.ones(g.number_of_edges())
        allo_CL_nojit = CL(g,initial_conductances,learning_rate=lr, jax = False)
        allo_CL = CL(g,initial_conductances,learning_rate=lr, jax = True)
        allo_GD = CL(g,initial_conductances,learning_rate=lr, jax = True)

        nodes_source = np.array([0,49,23])
        indices_source = np.array([list(allo_CL.graph.nodes).index(node) for node in nodes_source])
        inputs_source = np.array([0.,0.5,2.])

        nodes_target = np.array([15,43])
        indices_target = np.array([list(allo_CL.graph.nodes).index(node) for node in nodes_target])
        inputs_target = np.array([0.4,0.9])

        _ = allo_CL_nojit.set_task(indices_source, inputs_source, indices_target, inputs_target)
        _ = allo_CL.jax_set_task(indices_source, inputs_source, indices_target, inputs_target)
        _ = allo_GD.jax_set_task(indices_source, inputs_source, indices_target, inputs_target)

        n_epochs = 100
        n_steps_per_epoch = 10
        eta = 0.1

        start_time = time.time()
        _,_,_,_,_ = allo_CL_nojit.train(n_epochs, n_steps_per_epoch, eta = eta, verbose = False, pbar = False, log_spaced = False, save_state = False)
        end_time = time.time()
        training_time_CL_nojit = end_time - start_time

        start_time = time.time()
        _,_ = allo_CL.train_CL(n_epochs, n_steps_per_epoch, eta = eta, verbose = False, pbar = False, log_spaced = False, save_state = False)
        end_time = time.time()
        training_time_CL = end_time - start_time

        start_time = time.time()
        _,_ = allo_GD.train_GD(n_epochs, n_steps_per_epoch, verbose = False, pbar = False, log_spaced = False, save_state = False)
        end_time = time.time()
        training_time_GD = end_time - start_time

        # Get the network details
        num_nodes = g.number_of_nodes()
        num_edges = g.number_of_edges()

        # Create the row of the table
        table_row = f"{size[0]} x {size[1]} | {num_nodes} | {num_edges} | {training_time_CL_nojit:.4f} | {training_time_CL:.4f} | {training_time_GD:.4f}\n"

        # Append the row to the file
        with open('speed_results.txt', 'a') as f:
            f.write(table_row)



if __name__ == "__main__":
    generate_speed_report()
