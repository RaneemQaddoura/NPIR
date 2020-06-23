import pandas as pd 
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def run(results_directory, indexing_ratio, iterations, filenames, ev_measures):
    plt.ioff()
    fileResultsDetailsData = pd.read_csv(results_directory + '/experiment_avg.csv')

    for filename in filenames: 
        for measure in ev_measures:
            data = []    

            for IR in indexing_ratio:
                for i in iterations:
                    detailedData = fileResultsDetailsData[(fileResultsDetailsData["Dataset"] == filename) & (fileResultsDetailsData["IR"] == IR) & (fileResultsDetailsData["i"] == i)]
                    detailedData = detailedData[measure].values
                    data.append(detailedData[0])
            
            data = np.array(data).reshape(len(indexing_ratio),len(iterations))
            ax = plt.axes(projection='3d')
            surf = ax.plot_surface(np.array(indexing_ratio), np.array(iterations), data, cmap='Blues')
            plt.colorbar(surf, shrink=0.7, aspect=7)

            fig_name = results_directory + "/surface-" + filename.replace('.csv','') + "-" + measure + ".png"
            plt.savefig(fig_name)
            #plt.show()
            plt.clf()