""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                                                                        """
"""                                   NPIR                                 """
"""                     Nearest Point with Indexing Ratio                  """
"""                             @author: Raneem                            """
"""                                version 1.0                             """
"""                                                                        """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from scipy import spatial
from treelib import Tree

import sys
import numpy as np
import random
      
def NPIR(points_,k,IR,i): 
    """    
    This is the implementation of the NPIR algorithm
    
    Parameters
    ----------    
    points_ : str
        The attribute values of all the points
    k : int
        Number of clusters
    IR : float
        The indexing ratio, 0 <= IR <= 1
    i : int
        Number of iterations, i > 0
    
    Returns
    -------
    list
        labelsPred: the predicted values of the points
    """
    global sortedDistancesTree, points, nPoints, nValues, sortedDistancesTree
    points = points_
    #check if index probability between 0 and 1
    if IR > 1 or IR < 0:
        print("The value of the index probability must be between 0 and 1");
        sys.exit()
        
    init(k, IR)
    calculate(i)
    
    return labelsPred
    
def init(k,IR):
    """    
    Initializes the variables and data structures, 
    and creates the initial points and distance k-dimensional tree
    
    Parameters
    ---------- 
    IR : float
        The indexing ratio, 0 <= IR <= 1
    k : int
        Number of clusters
        
    Returns
    -------
    N/A
    """
    global nPoints, nValues, points, assignmentsTree, labelsPred, assignedPoints
    global remainingPoints, distanceVectorIndex
    global assignmentsTree, nAssignedPoints
    global nRemainingPoints, sortedDistancesTree, pointsNearestIndices, pointsNearestDistances
    global maxIndex, nElections, assignersIndices, assignersDistances
        
    
    #initialize variables and lists
    nPoints = len(points)
    nValues = len(points[0])
    nValues = nValues - 1 #Dimension value
    assignmentsTree = Tree()
    nAssignedPoints = 0 #Number of points that are already assigned by the algorithm to a certain cluster (initially equals 0)
    nRemainingPoints = nPoints #Number of points that are waiting to be assigned by the algorithm to a certain cluster (initially equals to the number of points)
    labelsPred = np.array([None] * nPoints) #List of predicted cluster value for each point (initially equals -1 for every point)
    assignedPoints = np.array([]) #list containing the points that are already assigned by the algorithm to a certain cluster
    remainingPoints = np.array(range(nPoints))
    distanceVectorIndex = np.array([2] * nPoints) #List of index value of the distance vector for each point (initially equals 2 for every point)
    maxIndex = IR * nPoints * nPoints #maximum index value: A stopping criteria for moving points between clusters
    nElections = 0 #Number of elections (initially equals 0 elections)
    assignersIndices = np.array([None] * nPoints)
    assignersDistances = np.array([float("inf")] * nPoints)
    pointsNearestIndices =  [[n] for n in range(nPoints)]
    pointsNearestDistances =  [[0] for n in range(nPoints)]

    #Generate dimensional tree for the points and thier distance vectors
    sortedDistancesTree = spatial.cKDTree(points)
    
    #Generate initial points from the remaining points list
    createInitialPoints(k)
    
def createInitialPoints(k):
    """
    The main method
    Assigns points to clusters until all points are assigned or maximum index value is reached
    
    Parameters
    ----------    
    k : int
        Number of clusters
        
    Returns
    -------
    N/A
    """
    global nRemainingPoints, assignmentsTree, remainingPoints
    assignmentsTree.create_node("root","root")
    for clusterNo in range(k):
        initialPointIndex = remainingPoints[random.randint(0,nRemainingPoints - 1)]
        assignerIndex = -1
        distance = float("inf")
        addPointToCluster(clusterNo,initialPointIndex,assignerIndex,distance)
    
def calculate(i):
    """
    The main method
    Assigns points to clusters until all points are assigned or maximum index value is reached
    
    Parameters
    ----------    
    i : int
        Number of iterations, i > 0
        
    Returns
    -------
    N/A
    """
    global distanceVectorIndex,nRemainingPoints,maxIndex,nElections
    global assignedPoints,points,nPoints,sortedDistancesTree

    #iterate until number of iterations is reached
    for iter in range(i):
        
        # Distance Vector Index initially equals 2 for every point at each iteration
        distanceVectorIndex = [2] * nPoints
        #No elections are still there for current iteration
        nElections = 0
        #Assign points to clusters until all points are assigned (no points in remaining points list)
        #or reassign points to clusters until max index is reached
        while nRemainingPoints > 0 or (nRemainingPoints == 0 and maxIndex > (nElections + nPoints)):

            #select a random point that is already assigned to a cluster
            electedIndex = getRandomAssignedPoint()
            
            #Increment number of elections
            nElections += 1 
                			  
            nearestIndex,nearestDist = getNearestPoint(electedIndex)
            
            if nearestIndex == None:
                break;
    
            distanceVectorIndex[electedIndex] += 1 #In all cases the current index of the distance array should be incremented
            
            #check if Nearest already assigned
            if isPointInCluster(nearestIndex):#The Nearest is already assigned to a cluster
                if not arePointsInSameCluster(electedIndex, nearestIndex): #Case 3: The Nearest is already assigned to a different cluster than the Elected
                    #check if the Nearest should move to the cluster of the Elected
                    if shouldPointMoveToNearerCluster(nearestIndex, nearestDist): #Case 3b: The distance between the Nearest and the Assigner is greater than the distance between the Nearest and the Elected
                        moveNearestToElectedCluster(electedIndex, nearestIndex, nearestDist) # The Nearest moves to the cluster of the elected
                else:
                    changeAssigner(nearestIndex, electedIndex, nearestDist)
            else: #Case 1: The Nearest is not yet assigned to a cluster
                addNearestToElectedCluster(electedIndex, nearestIndex, nearestDist); #Add the Nearest to the cluster of the Elected
                            
def getRandomAssignedPoint():
    """
    Election operation: selects a random point from the list of assigned points

    Parameters
    ----------    
    N/A
    
    Returns
    -------
    int
        The index of the random Elected
    """
    global assignedPoints,nAssignedPoints
    return int(assignedPoints[random.randint(0,nAssignedPoints - 1)])
    
def getNearestPoint(electedIndex):
    """
    Selection operation: Selects the Nearest index and distance according to 
    the current index of the distance vector for the Elected

    Parameters
    ----------    
    electedIndex : int
        The index of the Elected
    
    Returns
    -------
    int
        The index of the Nearest
    float
        The distance between the Nearest and the Elected
    """
    global points,distanceVectorIndex,sortedDistancesTree
    pt = points[electedIndex] #The Elected point
    dvi = distanceVectorIndex[electedIndex] #the distance vector index        
            
    #Check if distance vector index reached to the end of the vector
    if dvi > nPoints:
        return None,None
    
    #Get the nearest point index & distance according to the current index of the distance vector        
    # p = 2 is the euclidean distance, p=1 is the manhatin distance
    # k here is for how many nearest neighbors to return
    
    #distance,index = sortedDistancesTree.query(pt,p=2,k=[dvi]) 
    distance,index = getNearestIndexAndDistance(electedIndex, pt, dvi) 
    return index,distance

def getNearestIndexAndDistance(pointIndex, pt,dvi):
    """
    Returns the index and distance of the Nearest point according to the distance vector index

    Parameters
    ----------    
    pt : ndarray
        The point that we need to fing its Nearest
    dvi : int
        The distance vectoor index
    
    Returns
    -------
    int
        The index of the Nearest point
    float
        The distance between the point and the Nearest point
    """
        
    global pointsNearestIndices, pointsNearestDistances
    if dvi < len(pointsNearestIndices[pointIndex]):
        index = pointsNearestIndices[pointIndex][dvi]
        distance = pointsNearestDistances[pointIndex][dvi]
    else:
        nearestDist,nearestIndex = sortedDistancesTree.query(pt,p=2,k=[dvi]) 
        index = nearestIndex[0] #The Nearest index
        distance = nearestDist[0] #The distance between the Nearest and the Elected
        pointsNearestIndices[pointIndex].append(index)
        pointsNearestDistances[pointIndex].append(distance)
        
    # try to store them in an array for next iterations 
    #and when we want to get point and distance we just take it from the list generated
    # in this case we only need to call query for far points not reached in previous iterations
    return distance,index

         
def isPointInCluster(pointIndex):
    """
    Checks if point is already assigned to a cluster

    Parameters
    ----------    
    pointIndex : int
        The index of the point to be checked
    assignerIndex: int
        The index of the Assigner point for the point to be checked
    
    Returns
    -------
    bool
        true/false indicating if the point is already assigned to a cluster        
    """
    global labelsPred    
    return labelsPred[pointIndex] != None
    
def arePointsInSameCluster(pointIndex1, pointIndex2):
    """
    Checks if two points are assigned to same cluster

    Parameters
    ----------    
    pointIndex1 : int
        The index of the first point to be checked
    pointIndex2: int
        The index of the second point to be checked
    
    Returns
    -------
    bool
        true/false indicating if the two points are assigned to same cluster        
    """
    global labelsPred
    return labelsPred[pointIndex1] == labelsPred[pointIndex2]
    
def shouldPointMoveToNearerCluster(nearestIndex, nearestDist): #Check if the Nearest should move to the cluster of the Elected
    """
    Checks if the Nearest should move to the cluster of the Elected

    Parameters
    ----------    
    nearestIndex : int
        The index of the Nearest point
    nearestDist: float
        The distance between the Elected and the Nearest
    
    Returns
    -------
    bool
        true/false if the Nearest should move to the cluster of the Elected        
    """
    global assignersDistances,assignersIndices
    if assignersIndices[nearestIndex] == None: #No assigner for the Nearest
        return True #Nearest should move to the cluster of the Elected
    else:
        return assignersDistances[nearestIndex] > nearestDist #Nearest is closer to Elected than Assigner
    
def moveNearestToElectedCluster(electedIndex, nearestIndex, distance): # The Nearest should move to the cluster of the elected
    """
    Assignment operation: changes the cluster of the Nearest to the cluster of the Elected. 
    The Nearest is already assigned

    Parameters
    ----------    
    electedIndex : int
        The index of the Elected
    nearestIndex: int
        The index of the Nearest
    distance: float
        The distance between the Elected and the Nearest
    
    Returns
    -------
    N/A
    """
    global labelsPred,nAssignedPoints,nRemainingPoints
    clusterNo = labelsPred[electedIndex] 
    
    oldClusterNo = labelsPred[nearestIndex]
    oldAssignerIndex = assignersIndices[nearestIndex]
    
    # Change the cluster of the Nearest to the cluster of the Elected
    addPointToCluster(clusterNo,nearestIndex,electedIndex,distance)
    # This to ensure that no cluster is left empty. K is preserved
    # Add a new point to the empty cluster from the remaining points. if no remaining points still exist, we add the point from the already assigned points
    if oldAssignerIndex == -1: # If the Nearest and its children are alone in its old cluster
        initialPointIndex = findNewPointForAnEmptyCluster()
        addPointToCluster(oldClusterNo,initialPointIndex,-1,float("inf")) # add the random selected point to the cluster of the Nearest so that the cluster is not left empty
    
def findNewPointForAnEmptyCluster():
    """
    Creates the initial point to a certain cluster

    Parameters
    ----------    
    N/A
    
    Returns
    -------
    int
        The index of the initial random point
    """
    global nRemainingPoints,nAssignedPoints,labelsPred,remainingPoints
    if nRemainingPoints > 0: # Select random point from remaining 
        initialPointIndex = int(remainingPoints[random.randint(0,nRemainingPoints - 1)])
    else: # No remaining points left, we need to take from asssigned
        # This to ensure that we have selected a random point from assigned that is not alone in cluster
        selectRandomPoint = True
        while(selectRandomPoint):
            initialPointIndex = int(assignedPoints[random.randint(0,nAssignedPoints - 1)])
            if not assignersIndices[initialPointIndex] == -1:
                selectRandomPoint = False  
    return initialPointIndex
        
def addNearestToElectedCluster(electedIndex, nearestIndex, distance): # The Nearest is assigned to the cluster of the elected
    """
    Assignment operation: changes the cluster of the Nearest to the cluster of the Elected.
    The Nearest is not yet assigned

    Parameters
    ----------    
    electedIndex : int
        The index of the Elected
    nearestIndex: int
        The index of the Nearest
    distance: float
        The distance between the Elected and the Nearest
    
    Returns
    -------
    N/A
    """
    global labelsPred
    clusterNo = labelsPred[electedIndex]
    addPointToCluster(clusterNo,nearestIndex,electedIndex,distance)
    
def addPointToCluster(clusterNo,pointIndex,assignerIndex,assignerDistance):
    """
    Adds a point to a cluster

    Parameters
    ----------
    clusterNo : int
        The cluster where the point should be added
    pointIndex : int
        The index of the point to be added
    assignerIndex: int
        The index of the Assigner point for the point to be added
    assignerDistance: float
        The distance between the point and the Assigner
    Returns
    -------
    N/A
    """
    global remainingPoints,assignedPoints,nAssignedPoints,nRemainingPoints,assignersDistances,assignersIndices,labelsPred 
                    
    if assignersIndices[pointIndex] == None: # Point not yet assigned
        nAssignedPoints += 1
        nRemainingPoints -= 1    
        assignedPoints = np.append(assignedPoints, pointIndex)
        remainingPoints = np.setdiff1d(remainingPoints, pointIndex)
        
    labelsPred[pointIndex] = clusterNo
    assignersIndices[pointIndex] = assignerIndex
    assignersDistances[pointIndex] = assignerDistance
    
    if(assignmentsTree.contains(pointIndex)):
        children=assignmentsTree.subtree(pointIndex).all_nodes_itr()
        childrenIndices = np.array([child.identifier for child in children])
        labelsPred = np.array(labelsPred)
        labelsPred[childrenIndices] = clusterNo
        '''
        #print "children"
        for child in children:
            index = child.identifier
            #print index,
            if index == "root":
                index = -1
            labelsPred[index] = clusterNo
        #print()
        '''
    
    updateAssignmentsTree(pointIndex,assignerIndex)
    #testTree()

    
def changeAssigner(nearestIndex, electedIndex, nearestDist): #Check if the Nearest should move to the cluster of the Elected
    """
    Checks if the Elected should be the Assigner for the Nearest and change it accordingly

    Parameters
    ----------    
    nearestIndex : int
        The index of the Nearest point
    nearestDist: float
        The distance between the Elected and the Nearest        
    electedIndex : int
        The index of the Elected
        
    Returns
    -------
    N/A        
    """
    global assignersDistances,assignersIndices,assignmentsTree
    #Nearest closest to Elected than Assigner
    if nearestDist < assignersDistances[nearestIndex] and not assignmentsTree.subtree(nearestIndex).contains(electedIndex): 
        assignersDistances[nearestIndex] = nearestDist
        assignersIndices[nearestIndex] = electedIndex        
        updateAssignmentsTree(nearestIndex,electedIndex)
    
def updateAssignmentsTree(pointIndex,assignerIndex):
    """
    Updates the assignment tree. The assignment tree contains the points that 
    are already assigned and their assigners and children in a tree data structure

    Parameters
    ----------    
    pointIndex : int
        The index of the point to be updated/added
    assignerIndex: int
        The index of the Assigner point for the point to be updated/added
    
    Returns
    -------
    N/A
    """
    global nAssignedPoints,nRemainingPoints,assignedPoints
    
    if assignerIndex == -1:
        assignerIndex = "root"
    
    if assignmentsTree.contains(pointIndex): # Point already assigned 
        pointTree = assignmentsTree.subtree(pointIndex)
        assignmentsTree.remove_node(pointIndex)        
        assignmentsTree.paste(assignerIndex, pointTree)
                
    else:  
        assignmentsTree.create_node(pointIndex, pointIndex, parent=assignerIndex)        
                