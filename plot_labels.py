import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from itertools import cycle, islice    
warnings.simplefilter(action='ignore')

def run(results_directory, filenames, ev_measures):
    plt.ioff()
    fileResultsLablesData = pd.read_csv(results_directory + '/experiment_details_Labels.csv', skiprows=[0], header=None)
    fileResultsDetailsData = pd.read_csv(results_directory + '/experiment_details.csv')
    for filename in filenames:     
        data = np.genfromtxt('datasets/' + filename + '.csv', delimiter=",")
        for measure in ev_measures:
            detailedData = fileResultsDetailsData[(fileResultsDetailsData["Dataset"] == filename)]
            detailedData = detailedData[measure].values       
            index = np.argmax(detailedData)

            labelsData = fileResultsLablesData[fileResultsLablesData[0] == filename]
            labelsData = labelsData.drop([0,1,2], axis = 1) 
            labels = np.array(labelsData).tolist()[index]
            k = max(labels) + 1
            fig = plt.figure()      
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3',
                                              '#999999', '#e41a1c', '#dede00']),int(k))))
            plt.scatter(data[:, 0], data[:, 1], s=10, color=colors[labels])      
            fig.savefig(results_directory + "/plot-" + filename + '-best' + measure + '.png') 
            plt.clf()     

