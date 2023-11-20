import numpy as np
from tqdm import tqdm

def spread_state(pos_x, pos_y, state):
    """Compute the spreading of a state over the network.

    Parameters
    ----------
    pos_x : array_like
        The x-coordinates of the nodes.
    pos_y : array_like
        The y-coordinates of the nodes.
    state : array_like
        The state of the network.

    Returns
    -------
    float
        The spreading of the state over the network.
    """
    return np.dot(state, pos_x**2*state)+np.dot(state, pos_y**2*state)-np.dot(state, pos_x*state)**2-np.dot(state, pos_y*state)**2

def spread_basis(pos_x, pos_y, basis):
    """Compute the spreading of a basis over the network.

    Parameters
    ----------
    pos_x : array_like
        The x-coordinates of the nodes.
    pos_y : array_like
        The y-coordinates of the nodes.
    basis : array_like
        The basis of the network. Each column is a basis vector.

    Returns
    -------
    float
        The spreading of the basis over the network.
    """
    n = basis.shape[1]
    spreading = 0
    for i in range(n):
        spreading += spread_state(pos_x, pos_y, basis[:,i])
    return spreading/n

def position_operators(pos_x, pos_y, basis):
    """Generate the projected position operators, from the position of the nodes and the basis of eigenvectors.

    Parameters
    ----------
    pos_x : array_like
        The x-coordinates of the nodes.
    pos_y : array_like
        The y-coordinates of the nodes.
    basis : array_like
        The basis of the network. Each column is a basis vector.

    Returns
    -------
    op_x, op_y : array_like
        The projected position operators.
    """
    op_x = basis.T @ np.diag(pos_x) @ basis
    op_y = basis.T @ np.diag(pos_y) @ basis
    
    return op_x, op_y

def gradient_spread_basis(pos_x, pos_y, basis):
    """
    Compute the gradient of the spreading function in terms of the positions x and y and the basis of the eigenvectors.
    """
    # Generate the projected position operators X and Y
    X, Y = position_operators(pos_x, pos_y, basis)
    # Compute the diagonal elements of X and Y
    Xdiag = np.diag(X)
    Ydiag = np.diag(Y)
    # Compute the off-diagonal elements of X and Y
    Xoffdiag = X - np.diag(Xdiag)
    Yoffdiag = Y - np.diag(Ydiag)
    # Compute the gradients
    grad_X = Xoffdiag @ np.diag(Xdiag) - np.diag(Xdiag) @ Xoffdiag
    grad_Y = Yoffdiag @ np.diag(Ydiag) - np.diag(Ydiag) @ Yoffdiag
    # The gradient is the difference between the gradients of X and Y
    grad = 2 * (grad_X + grad_Y)
    return grad

def _GD_step(pos_x, pos_y, basis, step_size, with_noise=False, noise_amplitude=0.1):
    """
    One step in the gradient descent algorithm for the localization functional.
    """
    # Compute the gradient of the spreading function
    grad = gradient_spread_basis(pos_x, pos_y, basis)
    if with_noise:
        noise_matrix = np.random.normal(size=grad.shape, scale=noise_amplitude)
        noise_matrix = (noise_matrix + noise_matrix.T) / 2
        grad += noise_matrix
    # Compute the new basis vectors
    new_basis = basis - step_size * basis @ grad.T
    # Normalize the new basis vectors
    new_basis = new_basis / np.linalg.norm(new_basis, axis=0)
    return new_basis

def localize(pos_x, pos_y, basis, step_size, n_epochs, n_steps_per_epoch, with_noise=False, noise_amplitude=0.1, pbar = True):
    """
    Iterate the gradient descent algorithm for the localization functional.
    """
    if pbar:
            epochs = tqdm(range(n_epochs))

    spread_history = []
    end_epoch = np.arange(1,n_epochs+1) * n_steps_per_epoch
    for epoch in epochs:
        for j in range(n_steps_per_epoch):
            basis = _GD_step(pos_x, pos_y, basis, step_size, with_noise, noise_amplitude)
        spread_history.append(spread_basis(pos_x, pos_y, basis))
    return basis, spread_history, end_epoch
