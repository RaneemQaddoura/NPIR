import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def run(results_directory, indexing_ratio, iterations, filenames, ev_measures):
    plt.ioff()
    
    fileResultsDetailsData = pd.read_csv(results_directory + '/experiment_details.csv')
    for filename in filenames:        
        for measure in ev_measures:
            data = []      
            for IR in indexing_ratio:
                for i in iterations:
                    
                    detailedData = fileResultsDetailsData[(fileResultsDetailsData["Dataset"] == filename) & (fileResultsDetailsData["IR"] == IR) & (fileResultsDetailsData["i"] == i)]
                    detailedData = detailedData[measure]
                    detailedData = np.array(detailedData).T.tolist()
                    data.append(detailedData)

            labels = [str(ir)+','+str(i) for ir in indexing_ratio for i in iterations]
            labels_legend = ['IR:'+str(ir)+', i:'+str(i) for ir in indexing_ratio for i in iterations]
            box=plt.boxplot(data,patch_artist=True, labels=labels)
            

            colors = ['#5c9eb7','#f77199', '#cf81d2','#4a5e6a','#f45b18',
            '#ffbd35','#6ba5a1','#fcd1a1','#c3ffc1','#68549d',
            '#1c8c44','#a44c40','#404636']
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
             
            plt.legend(handles= box['boxes'], labels=labels_legend, 
                    loc="upper right", bbox_to_anchor=(1.3,1.02))
            fig_name = results_directory + "/boxplot-" + filename + "-" + measure + ".png"
            plt.savefig(fig_name, bbox_inches='tight')
            plt.clf()
            #plt.show()
        


