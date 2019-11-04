from sklearn import metrics
from NPIR import NPIR

import numpy as np
import datetime
import warnings
import statistics as stat
import os
#import cProfile

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def readDataset(directory, filename):
    """
    Reads the dataset file and store a list of Point

    Parameters
    ----------
    directory : str
        The file location of the dataset
    filename : str
        The dataset file name

    Returns
    -------
    list
        a list of points containing the features for every point
    """
    my_path = os.path.abspath(os.path.dirname(__file__))
    raw_data = open(os.path.join(my_path, directory + filename), 'rt')
    data = np.loadtxt(raw_data, delimiter=",")
    return data


###Evaluation
directory = ""
 

filenames = ["VaryDensity.csv","jain.csv"]
IR_ = [0.15, 0.15]#IR: The indexing ratio to be used for generating the maximum index
i_ = [30, 50]#The number of iteration i

runs = 30


for d in range(len(filenames)):
    filename = filenames[d]
    IR = IR_[d]
    i = i_[d]
    HS = [0]*runs
    CS = [0]*runs
    VM = [0]*runs
    AMI = [0]*runs
    ARI = [0]*runs
    Aggregate = [0]*runs
    
    print()
    print("dataset," + filename)
    print("IR," + str(IR))
    print("i," + str(i))
    print("run,homo,compl,vmeasure,AMI,ARI,Aggregate,time")
    
    for run in range(runs):
        
        #Get current time (start time)
        format = '%m/%d/%y %H:%M:%S'
        tStart = datetime.datetime.now()
        
        #Read dataset
        data = readDataset(directory, filename) #Data containing points with their features and true values
        points = data[:,:-1] #list of points
        k = len(np.unique(data[:,-1]))#k: Number of clusters
        labelsTrue = data[:,-1] #List of actual cluster of each points (last field)
    
        labelsPred = NPIR(points,k,IR,i)
        
        tEnd = datetime.datetime.now()
        HS[run] = float("%0.2f"%metrics.homogeneity_score(labelsTrue,labelsPred))
        CS[run] = float("%0.2f"%metrics.completeness_score(labelsTrue,labelsPred))
        VM[run] = float("%0.2f"%metrics.v_measure_score(labelsTrue,labelsPred))
        AMI[run] = float("%0.2f"%metrics.adjusted_mutual_info_score(labelsTrue,labelsPred))
        ARI[run] = float("%0.2f"%metrics.adjusted_rand_score(labelsTrue,labelsPred))
        Aggregate[run] = float("%0.2f"%(float("%0.2f"%(HS[run] + CS[run] + VM[run] + AMI[run] + ARI[run])) / 5))
        print(str(run + 1) +"," + str(HS[run]) +"," +str(CS[run]) +"," +str(VM[run]) +"," +str(AMI[run]) +"," +str(ARI[run]) +"," +str(Aggregate[run]) +"," +str(tEnd - tStart))
    
    print("AVG," + str(float("%0.2f"%(sum(HS)/runs)))+"," 
                       +str(float("%0.2f"%(sum(CS)/runs))) +"," 
                            +str(float("%0.2f"%(sum(VM)/runs))) +"," 
                                 +str(float("%0.2f"%(sum(AMI)/runs))) +"," 
                                     +str(float("%0.2f"%(sum(ARI)/runs))) +"," 
                                         +str(float("%0.2f"%(sum(Aggregate)/runs))))
    print("Best," + str(float("%0.2f"%(max(HS))))+"," 
                       +str(float("%0.2f"%(max(CS)))) +"," 
                            +str(float("%0.2f"%(max(VM)))) +"," 
                                 +str(float("%0.2f"%(max(AMI)))) +"," 
                                     +str(float("%0.2f"%(max(ARI)))) +"," 
                                         +str(float("%0.2f"%(max(Aggregate)))))      
    print("STD," + str(float("%0.2f"%(stat.stdev(HS)))) + "," 
                        +str(float("%0.2f"%(stat.stdev(CS)))) +"," 
                             +str(float("%0.2f"%(stat.stdev(VM)))) +"," 
                                  +str(float("%0.2f"%(stat.stdev(AMI)))) +"," 
                                      +str(float("%0.2f"%(stat.stdev(ARI)))) +"," 
                                          +str(float("%0.2f"%(stat.stdev(Aggregate)))))    
