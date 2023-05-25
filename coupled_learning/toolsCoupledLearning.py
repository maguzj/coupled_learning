############################ COUPLED LEARNING FUNCTIONS ###############################

import numpy as np
from numpy import zeros, ones, diag, array, where, dot, c_, r_, arange, sum


from scipy.linalg import solve
from scipy.sparse import csr_matrix
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve


def constraintMatrix(NodeData, Nodes, EdgeData, Edges, GroundNodes, NN, EI, EJ):
    '''
    TODO: 
    1. Rename variables to more explicit ones.
    2. Generalize NodeData to include ground.
    3. Set GroundNodes as an optional argument.
    4. Understand the Edges and EdgeData variables.


    Input:
        NodeData: physical values (voltages) imposed at the nodes Nodes
        Nodes: indices of nodes at which we impose a given physical value
        EdgeData: ?????
        Edges: List of ?????
        GroundNodes: indices of nodes at which we impose the ground values (set to 0)
        NN: Number of nodes
        EI: List of nodes_i
        EJ: List of nodes_j
    Output:
        constraintM:    Constraint matrix (constraintM)linked to both ground nodes and source nodes.
                        constraintM is rectangular matrix of size (#GroundNodes + #SourceNodes) x (#Nodes+1).
                        The first block (#GroundNodes x #Nodes) is a sparse matrix with 1 entries
                        at the columns associated to the ground nodes.
                        The last column corresponds to the physical values (NodeData) assigned to
                        the ground nodes, often set to 0.
    '''
    # Constraints due to the ground nodes
    nGround = len(GroundNodes)
    indicesGround = arange(nGround)
    constraintM = zeros([nGround, NN+1])
    constraintM[indicesGround, GroundNodes] = +1.
    constraintM[:, NN] = 0.
    
    # Constraints due to the imposed nodes
    if len(Nodes):
        nImpNodes = len(Nodes)
        indicesImpNodes = arange(nImpNodes)
        constraintImpNodesM = zeros([nImpNodes, NN+1])
        constraintImpNodesM[indicesImpNodes, Nodes] = +1.
        constraintImpNodesM[:, NN] = NodeData
        constraintM = np.concatenate((constraintM, constraintImpNodesM))
    
    # Constraints due to the imposed edges
    # I have yet to understand this part. I will comment it for now.
    # if len(Edges):
    #     nImpEdges = len(Edges)
    #     indicesImpEdges = arange(nImpEdges)
    #     constraintImpEdgesM = zeros([nImpEdges, NN+1])
    #     constraintImpEdgesM[indicesImpEdges, EI[Edges]] = +1.
    #     constraintImpEdgesM[indicesImpEdges, EJ[Edges]] = -1.
    #     constraintImpEdgesM[:, NN] = EdgeData
    #     constraintM = np.concatenate((constraintM, constraintImpEdgesM))
    return constraintM



def linearEqSolution(conductanceArray, constraintMatrix, incidenceMatrix, NN):
    hessian = dot(incidenceMatrix.T,incidenceMatrix*(conductanceArray.reshape(len(conductanceArray),1)))
    nConstraints = len(constraintMatrix)
    extendedLaplacian = np.concatenate((np.concatenate((hessian,constraintMatrix[:,:NN])),np.concatenate((constraintMatrix[:,:NN].T,zeros([nConstraints,nConstraints])))),axis=1)

    f = zeros(NN+nConstraints)
    f[NN:] = constraintMatrix[:,NN]
    P = solve(extendedLaplacian, f, assume_a='sym', check_finite=False)[:NN]
    return P



def cost(K, constraintMatrix, constraintTargetMatrix, incidenceMatrix, NN, rel=0, weights=0):
    # Returns the L2 cost functions
    # rel potentially weights different tasks with a different weight. Default is uniform weight
    P = linearEqSolution(K, constraintMatrix, incidenceMatrix, NN) # Computes pressure or voltage at the nodes
    FSR = dot(constraintTargetMatrix[:,:NN], P) # pick the response at the target nodes
    Errors = 0.5 * (FSR - constraintTargetMatrix[:,NN]) ** 2.
    if rel != 0:
        Errors *= weights
    return sum(Errors)

def loss(freeStateVoltages, targetVoltages):
    '''
    returns the L2 loss function
    '''
    return 0.5*sum((freeStateVoltages-targetVoltages)**2)

def lossArray(Tasks):
    '''
    input:
        Tasks: list of pair tasks (freeState, target)
    output:
        lossArray: list of L2 loss functions per each pair task in Tasks.
    '''
    return array([loss(task[0],task[1]) for task in Tasks])


def costArray(K, pairTasks, incidenceMatrix, NN, rel=0, weights=0):
    # task[0] is the constraint matrix for the free state
    # task[1] is the constraint matrix for the target nodes only (with neither sources nor groundnodes)
    return array([cost(K, task[0], task[1], incidenceMatrix, NN, rel, weights) for task in pairTasks])


def freeStates(K, Tasks, incidenceMatrix, NN):
    freeStateP = array([linearEqSolution(K, task[0], incidenceMatrix, NN) for task in Tasks])
    return freeStateP


def clampedStates(freeStateP, K, Tasks, incidenceMatrix, NN, eta=1.e-3):
    TasksNudge = []
    clampedStateP = []
    
    for i in range(len(Tasks)):
        task = Tasks[i]
        sourceConstraintMatrix = task[0]
        targetConstraintMatrix = task[1]
  
        freeStatePTargetNodes = dot(targetConstraintMatrix[:,:NN], freeStateP[i])
        
        Nudge = targetConstraintMatrix.copy()
        Nudge[:,NN] = freeStatePTargetNodes + eta * (targetConstraintMatrix[:,NN] - freeStatePTargetNodes)   
        clampedStateP.append(linearEqSolution(K, np.vstack([sourceConstraintMatrix, Nudge]), incidenceMatrix, NN))
        
    clampedStateP = array(clampedStateP)
    return clampedStateP


####################################################################################################################### SPARSE IMPLEMENTATION #######################################################################################################################


def sparseConstraintQMatrix(indicesNodes, NN):
    '''
    Input:
        indicesNodes: Array of indices at which we impose a given physical value.
        NN: total number of nodes in the network.
    Output:
        sparseQmatrix: sparseQmatrix (Q) is a sparse constraint rectangular matrix of size NNxlen(indicesNodes). Its entries are only 1 or 0.
                       Q.Q^T is a projector onto to the space of the nodes in nodeArray.
    '''
    # Check indicesNodes is a non-empty array
    if len(indicesNodes) == 0:
        raise ValueError('indicesNodes must be a non-empty array.')
    # Create the sparse rectangular constraint matrix Q using csr_matrix. Q has entries 1 at the indicesNodes[i] row and i column.
    sparseQmatrix = csr_matrix((ones(len(indicesNodes)), (indicesNodes, arange(len(indicesNodes)))), shape=(NN, len(indicesNodes)))
    return sparseQmatrix


def sparseHessian(conductanceArray, incidenceMatrix):
    '''
    Input:
        conductanceArray: Array of conductances at each edge.
        incidenceMatrix: Incidence matrix of the network.
    Output:
        sparseHessian: sparseHessian is a sparse matrix of size NNxNN.
    '''
    # Check that the incidence matrix is a sparse matrix, otherwise raise a warning.
    if not isinstance(incidenceMatrix, csr_matrix):
        print('Warning: the incidence matrix is not in sparse format. Transformations between dense and sparse are costly.')
        sparseHessian = csr_matrix(incidenceMatrix.T.dot(incidenceMatrix*conductanceArray.reshape(len(conductanceArray),1)))
    else:
        sparseHessian = incidenceMatrix.T.dot(incidenceMatrix.multiply(conductanceArray.reshape(len(conductanceArray),1)))
    
    return sparseHessian

def sparseLinearEqSolution(sparseHessian, indicesNodes, imposedVoltageArray):
    '''
    Input:
        sparseHessian: sparse matrix of size NNxNN.
        indicesNodes: Array of indices at which we impose a given physical value.
        imposedVoltageArray: Array of imposed physical values at the indicesNodes.
    Output:
        P: Array of voltages at the nodes.
    '''
    # determine the number of nodes from the sparseHessian shape
    NN = sparseHessian.shape[0]
    # create the sparse constraint matrix Q
    sparseQmatrix = sparseConstraintQMatrix(indicesNodes, NN)
    # create the extended hessian as a block matrix [[H, Q], [Q.T, 0]]
    sparseExtendedHessian = bmat([[sparseHessian, sparseQmatrix], [sparseQmatrix.T, None]], format='csr', dtype=float)
    # create the extended right hand side as a block vector [0, imposedVoltageArray]
    extendedRightHandSide = np.hstack([zeros(NN), imposedVoltageArray])
    # solve the linear system
    P = spsolve(sparseExtendedHessian, extendedRightHandSide)[:NN]
    return P

def sparseLinearEqSolution_directQmatrix(sparseHessian, sparseQmatrix, extendedRHSimposedVoltageArray):
    '''
    Input:
        sparseHessian: sparse matrix of size NNxNN.
        sparseQmatrix: sparse matrix of size NNxlen(indicesNodes).
        extendedRHSimposedVoltageArray: zero array stacked with the array of imposed physical values at the indicesNodes.
    Output:
        P: Array of voltages at the nodes.
    '''
    # # determine the number of nodes from the sparseHessian shape
    NN = sparseHessian.shape[0]
    sparseExtendedHessian = bmat([[sparseHessian, sparseQmatrix], [sparseQmatrix.T, None]], format='csr', dtype=float)
    # solve the linear system
    P = spsolve(sparseExtendedHessian, extendedRHSimposedVoltageArray)[:NN]
    return P

def clampFreeState(freeState,eta,voltageTargetArray,indicesTargetArray):
    '''
    Input:
        freeState: Array of voltages at the nodes.
        eta: small number to nudge the clamped state.
        voltageTargetArray: Array of target voltages at the indicesTargetArray.
        indicesTargetArray: Array of indices at which we impose a given target voltage.
    Output:
        clampedState: state with clamped voltages at the target nodes.
    '''
    clampedState = freeState.copy()
    clampedState[indicesTargetArray] = freeState[indicesTargetArray] + eta * (voltageTargetArray - freeState[indicesTargetArray])
    return clampedState



def stepSparseCoupledLearning(sparseIncidenceMatrix, conductances, indicesImposedNodesFreeState, indicesTargets, indicesImposedNodesClampedState, freeStateImposedVoltage,desiredTargetVoltages,eta, alpha, kmin, kmax):
    '''
    Performs one step in the coupled learning algorithm using sparse matrices
    Input:
        sparseIncidenceMatrix: sparse incidence matrix of the network.
        conductances: Array of conductances at each edge.
        indicesImposedNodesFreeState: Array of indices at which we impose a given physical value for the free state.
        indicesTargets: Array of indices at which we impose a given target voltage.
        indicesImposedNodesClampedState: Array of indices at which we impose a given physical value for the clamped state.
        freeStateImposedVoltage: Array of imposed physical values at the indicesImposedNodesFreeState.
        clampedStateImposedVoltage: Array of imposed physical values at the indicesImposedNodesClampedState.
        eta: small number to nudge the clamped state.
        alpha: learning rate.
        kmin: minimum conductance.
        kmax: maximum conductance.
    Output:
        updatedConductances: Array of conductances at each edge.
    '''
    # compute the sparse hessian
    sHessian = sparseHessian(conductances, sparseIncidenceMatrix)
    # compute the free state and the voltage drop associated to each edge
    freeState = sparseLinearEqSolution(sHessian, indicesImposedNodesFreeState, freeStateImposedVoltage)
    voltageDrop = sparseIncidenceMatrix.dot(freeState)
    # define the nudge
    nudge = freeState[indicesTargets] + eta * (desiredTargetVoltages - freeState[indicesTargets])
    clampedStateImposedVoltage = np.concatenate((freeStateImposedVoltage, nudge))
    # compute the clamped state and the voltage drop associated to each edge
    clampedState = sparseLinearEqSolution(sHessian, indicesImposedNodesClampedState, clampedStateImposedVoltage)
    voltageDropClamped = sparseIncidenceMatrix.dot(clampedState)
    # compute delta conductances (-gradient of the cost function)
    deltaConductances = -alpha/eta*(voltageDropClamped**2 - voltageDrop**2)
    # update the conductances
    updatedConductances = conductances + deltaConductances
    # clip the conductances
    updatedConductances = np.clip(updatedConductances, kmin, kmax)
    # return the free state, the voltage drop, the delta conductances and the updated conductances
    return freeState, voltageDrop, deltaConductances, updatedConductances

def stepSparseCoupledLearning_directQmatrix(sparseIncidenceMatrix, conductances, sparseQMatrixFree, indicesTargets, sparseQMatrixClamped, extendedRHSFree,desiredTargetVoltages,eta, alpha, kmin, kmax):
    '''
    Performs one step in the coupled learning algorithm using sparse matrices
    Input:
        sparseIncidenceMatrix: sparse incidence matrix of the network.
        conductances: Array of conductances at each edge.
        sparseQMatrixFree: sparse matrix of size NNxlen(indicesImposedNodesFreeState).
        indicesTargets: Array of indices at which we impose a given target voltage.
        sparseQMatrixClamped: sparse matrix of size NNxlen(indicesImposedNodesClampedState).
        extendedRHSFree: zero array stacked with the array of imposed physical values at the indicesImposedNodesFreeState.
        desiredTargetVoltages: Array of target voltages at the indicesTargets.
        eta: small number to nudge the clamped state.
        alpha: learning rate.
        kmin: minimum conductance.
        kmax: maximum conductance.
    Output:
        freeState: Array of voltages at the nodes.
        voltageDrop: Array of voltage drops at each edge.
        deltaConductances: Array of delta conductances at each edge.
        updatedConductances: Array of conductances at each edge.
    '''
    # compute the sparse hessian
    sHessian = sparseHessian(conductances, sparseIncidenceMatrix)
    # compute the free state and the voltage drop associated to each edge
    freeState = sparseLinearEqSolution_directQmatrix(sHessian, sparseQMatrixFree, extendedRHSFree)
    voltageDrop = sparseIncidenceMatrix.dot(freeState)
    # define the nudge
    nudge = freeState[indicesTargets] + eta * (desiredTargetVoltages - freeState[indicesTargets])
    extendedRHSClamped = np.concatenate((extendedRHSFree, nudge))
    # compute the clamped state and the voltage drop associated to each edge
    clampedState = sparseLinearEqSolution_directQmatrix(sHessian, sparseQMatrixClamped, extendedRHSClamped)
    voltageDropClamped = sparseIncidenceMatrix.dot(clampedState)
    # compute delta conductances (-gradient of the cost function)
    deltaConductances = -alpha/eta*(voltageDropClamped**2 - voltageDrop**2)
    # update the conductances
    updatedConductances = conductances + deltaConductances
    # clip the conductances
    updatedConductances = np.clip(updatedConductances, kmin, kmax)
    # return the free state, the voltage drop, the delta conductances and the updated conductances
    return freeState, voltageDrop, deltaConductances, updatedConductances

def loopSparseCoupledLearning(sparseIncidenceMatrix, conductances, indicesImposedNodesFreeState, indicesTargets, indicesImposedNodesClampedState, freeStateImposedVoltage,desiredTargetVoltages,eta, alpha, kmin, kmax, nIterations):
    ''''
    Performs nIterations steps in the coupled learning algorithm using sparse matrices
    '''
    for i in range(nIterations):
        freeState, voltageDrop, deltaConductances, conductances = stepSparseCoupledLearning(sparseIncidenceMatrix, conductances, indicesImposedNodesFreeState, indicesTargets, indicesImposedNodesClampedState, freeStateImposedVoltage,desiredTargetVoltages,eta, alpha, kmin, kmax)
    return freeState, voltageDrop, deltaConductances, conductances

def loopSparseCoupledLearning_directQmatrix(sparseIncidenceMatrix, conductances, sparseQMatrixFree, indicesTargets, sparseQMatrixClamped, extendedRHSFree,desiredTargetVoltages,eta, alpha, kmin, kmax, nIterations):
    ''''
    Performs nIterations steps in the coupled learning algorithm using sparse matrices
    '''
    for i in range(nIterations):
        freeState, voltageDrop, deltaConductances, conductances = stepSparseCoupledLearning_directQmatrix(sparseIncidenceMatrix, conductances, sparseQMatrixFree, indicesTargets, sparseQMatrixClamped, extendedRHSFree,desiredTargetVoltages,eta, alpha, kmin, kmax)
    return freeState, voltageDrop, deltaConductances, conductances
