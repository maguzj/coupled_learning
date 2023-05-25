import numpy as np
from toolsCoupledLearning import *
import itertools
from scipy import sparse

class Network:
    def __init__(self, nodeArray, EI, EJ):
        self.nodeArray = nodeArray
        self.EI = np.array(EI)
        self.EJ = np.array(EJ)
        self.NE = len(EI)
        self.NN = len(nodeArray)

        incidenceMatrix = np.zeros([len(EI),len(nodeArray)])
        for i,(ei,ej) in enumerate(zip(EI,EJ)):
            i1 = np.where(nodeArray == ei)
            i2 = np.where(nodeArray == ej)
            incidenceMatrix[i,i1] = 1
            incidenceMatrix[i,i2] = -1

        self.incidenceMatrix = incidenceMatrix
        self.sparseIncidenceMatrix = sparse.csr_matrix(incidenceMatrix)

    def setPositions(self, positions):
        # positions is a list of tuples
        assert len(positions) == self.NN, 'positions must have the same length as the number of nodes'
        self.positions = positions
        
    
    def printNetwork(self):
        pass

    def setSourcesGroundsAndTargets(self, nSources, nGrounds, nTargets): 
        SpecialNodes = np.random.choice(range(self.NN), size=nSources+nTargets+nGrounds, replace=False)
        self.sourceNodes = SpecialNodes[:nSources]
        self.targetNodes = SpecialNodes[nSources:nSources+nTargets]
        self.groundNodes = [SpecialNodes[-nGrounds]]
        self.hiddenNodes = list(set(range(self.NN)) - set(SpecialNodes))

        # indices for the special nodes. In the case of a full grid they must coincide with sourceNodes, targetNodes, and groundNodes
        self.indicesSourceNodes = np.where(self.nodeArray == np.array(self.sourceNodes)[:,None])[1]
        self.indicesGroundNodes = np.where(self.nodeArray == np.array(self.groundNodes)[:,None])[1]
        self.indicesTargetNodes = np.where(self.nodeArray == np.array(self.targetNodes)[:,None])[1]
    
    def setSourcesGroundsAndTargets_directValues(self, sourceList, groundList, targetList): 
        # check not repeated indices
        assert len(set(sourceList) | set(groundList) | set(targetList)) == len(sourceList) + len(groundList) + len(targetList), 'source, ground, targets must not have overlapping elements'

        self.sourceNodes = sourceList #SpecialNodes[:nSources]
        self.targetNodes = targetList #SpecialNodes[nSources:nSources+nTargets]
        self.groundNodes = groundList
        self.hiddenNodes = list(set(range(self.NN)) - set(self.sourceNodes) - set(self.targetNodes) - set(self.groundNodes))

        # indices for the special nodes. In the case of a full grid they must coincide with sourceNodes, targetNodes, and groundNodes
        self.indicesSourceNodes = np.where(self.nodeArray == np.array(self.sourceNodes)[:,None])[1]
        self.indicesGroundNodes = np.where(self.nodeArray == np.array(self.groundNodes)[:,None])[1]
        self.indicesTargetNodes = np.where(self.nodeArray == np.array(self.targetNodes)[:,None])[1]

    def getSourceAndGroundContraintMatrix(self, sourceNodesValues):
        # ground nodes set to zero from the function constraintMatrix
        # indicesSourceNodes = np.where(self.nodeArray == np.array(self.sourceNodes)[:,None])[1]
        # indicesGroundNodes = np.where(self.nodeArray == np.array(self.groundNodes)[:,None])[1]
        return constraintMatrix(sourceNodesValues, self.indicesSourceNodes, [], [], self.indicesGroundNodes, self.NN, self.EI, self.EJ)
        #return constraintMatrix(sourceNodesValues, self.sourceNodes, [], [], self.groundNodes, self.NN, self.EI, self.EJ)

    def getTargetConstraintMatrix(self, sourceNodesValues):
        # indicesTargetNodes = np.where(self.nodeArray == np.array(self.targetNodes)[:,None])[1]
        return constraintMatrix(sourceNodesValues, self.indicesTargetNodes, [], [], [], self.NN, self.EI, self.EJ)
        # return constraintMatrix(sourceNodesValues, self.targetNodes, [], [], [], self.NN, self.EI, self.EJ)

        



class gridNetwork(Network):
    def __init__(self, size, interactionRange=1):
        # construct network
        a, b = size
        NN = a*b
        xs = np.arange(a)
        ys = np.arange(b)
        positionsList = [*itertools.product(*[xs,ys])]
        EI = []  # edge_i
        EJ = []  # edge_j

        # Define nearest neighbors
        indexNode = 0
        while indexNode < NN:
            pos = positionsList[indexNode]
            if pos[1] < b-1:
                EI.append(indexNode)
                EJ.append(indexNode+1)
            if pos[0] < a-1:
                EI.append(indexNode)
                EJ.append(indexNode+b)
            indexNode += 1
        NE1 = len(EI)

        # Define next-nearest neighbors
        indexNode = 0
        if interactionRange > 1:
            while indexNode < NN:
                pos = positionsList[indexNode]
                if (pos[0] < a-1) and (pos[1] < b-1):
                    EI.append(indexNode)
                    EJ.append(indexNode+b+1)
                if (pos[0] > 0) and (pos[1] < b - 1):
                    EI.append(indexNode)
                    EJ.append(indexNode-b+1)
                indexNode += 1
        NE2 = len(EI) - NE1

        # Define next-next-nearest neighbors
        indexNode = 0
        if interactionRange > 2:
            while indexNode < NN:
                pos = positionsList[indexNode]
                if pos[1] < b-2:
                    EI.append(indexNode)
                    EJ.append(indexNode+2)
                if pos[0] < a-2:
                    EI.append(indexNode)
                    EJ.append(indexNode+2*b)
                indexNode += 1
        NE3 = len(EI) - NE1 - NE2

        incidenceMatrix = np.zeros([NE1,NN])
        for i in range(NE1):
            incidenceMatrix[i,EI[i]] = +1
            incidenceMatrix[i,EJ[i]] = -1
        
        #self.EI = np.array(EI)
        #self.EJ = np.array(EJ)
        #self.NE = NE1
        #self.NN = NN
        #self.incidenceMatrix = incidenceMatrix
        nodeArray = np.arange(NN)

        super().__init__(nodeArray, EI, EJ)
        self.size = size
        self.interactionRange = interactionRange
        self.positions = positionsList