from sklearn import metrics
from NPIR import NPIR
import numpy as np
import datetime
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Select data sets
#"aggregation","aniso","appendicitis","blobs","circles","diagnosis_II","flame","glass","iris","iris2D","jain","liver","moons","mouse","pathbased","seeds","smiley","varied","vary-density","wdbc","wine"
filename = "datasets/vary-density.csv"
#IR: The indexing ratio to be used for generating the maximum index
IR = 0.15
#The number of iteration i
i = 50
 
#Get current time (start time)
tStart = datetime.datetime.now()

#data manipulation
data = np.genfromtxt(filename, delimiter=',')
points = data[:,:-1] #list of points
k = len(np.unique(data[:,-1]))#k: Number of clusters
labelsTrue = data[:,-1] #List of true class of each points (last field)

#running NPIR
labelsPred = NPIR(points,k,IR,i)

#printing results
tEnd = datetime.datetime.now()
print('Time: ' + str(tEnd - tStart))
print('labels:')
print(labelsPred)
print('Measures:')
print('HS: ' + str(metrics.homogeneity_score(labelsTrue,labelsPred)))
print('CS: ' + str(metrics.completeness_score(labelsTrue,labelsPred)))
print('VM: ' + str(metrics.v_measure_score(labelsTrue,labelsPred)))
print('AMI: ' + str(metrics.adjusted_mutual_info_score(labelsTrue,labelsPred)))
print('ARI: ' + str(metrics.adjusted_rand_score(labelsTrue,labelsPred)))