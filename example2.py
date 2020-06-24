from sklearn import metrics
from NPIR import NPIR
import numpy as np
import datetime
import warnings
import statistics as stat
import time
import csv
import plot_boxplot as box_plot
import plot_surface as surface
from pathlib import Path
warnings.simplefilter(action='ignore', category=FutureWarning)

#directories
datasets_directory = "datasets/"
results_directory = time.strftime("%Y-%m-%d-%H-%M-%S") + '/'
Path(results_directory).mkdir(parents=True, exist_ok=True)
 
# Select data sets
#"aggregation","aniso","appendicitis","blobs","circles","diagnosis_II","flame","glass","iris","iris2D","jain","liver","moons","mouse","pathbased","seeds","smiley","varied","vary-density","wdbc","wine"
filenames = ["iris2D","vary-density"]
#IR: The indexing ratio to be used for generating the maximum index
indexing_ratio = [0.1, 0.15]
#The number of iteration i
iterations = [30, 50]
#Choose whether to Export the results in different formats
# The length of indexing_ratio & iterations lists should be at least 3 for Export_surface to be set as True
export_flags = {'Export_avg':True, 'Export_best':True, 'Export_details':True, 
'Export_details_labels':True, 'Export_boxplot':True, 'Export_surface':True, 'Export_plot_labels':True}
#Select number of runs
NumOfRuns = 5

Flag_avg=False
Flag_best=False
Flag_details=False
Flag_details_Labels=False

for filename in filenames:
	for IR in indexing_ratio:
		for i in iterations:
		    HS = [0]*NumOfRuns
		    CS = [0]*NumOfRuns
		    VM = [0]*NumOfRuns
		    AMI = [0]*NumOfRuns
		    ARI = [0]*NumOfRuns
		    
		    print()
		    print("dataset," + filename)
		    print("IR," + str(IR))
		    print("i," + str(i))
		    print("run,homo,compl,vmeasure,AMI,ARI,time")
		    
		    for run in range(NumOfRuns):
		        
		        #Get current time (start time)
		        format = '%m/%d/%y %H:%M:%S'
		        tStart = datetime.datetime.now()
		        
		        data = np.genfromtxt(datasets_directory + filename + '.csv', delimiter=',')
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
		        print(str(run + 1) +"," + str(HS[run]) +"," +str(CS[run]) +"," +str(VM[run]) +"," +str(AMI[run]) +"," +str(ARI[run]) +"," +str(tEnd - tStart))

		        if(export_flags['Export_details_labels']==True):
		        	ExportToFileDetailsLabels=results_directory + "experiment_details_Labels.csv"
		        	with open(ExportToFileDetailsLabels, 'a',newline='\n') as out_details_labels:
		        		writer_details = csv.writer(out_details_labels,delimiter=',')
		        		if (Flag_details_Labels==False): # just one time to write the header of the CSV file
		        			header_details= np.concatenate([["Dataset", "IR","i"]])
		        			writer_details.writerow(header_details)
		        			Flag_details_Labels = True
		        		a=np.concatenate([[filename, IR, i], labelsPred])  
		        		writer_details.writerow(a)
		        	out_details_labels.close() 

		        if(export_flags['Export_details']==True):
		        	ExportToFileDetails=results_directory + "experiment_details.csv"
		        	with open(ExportToFileDetails, 'a',newline='\n') as out_details:
		        		writer_details = csv.writer(out_details,delimiter=',')
		        		if (Flag_details==False): # just one time to write the header of the CSV file
		        			header_details= np.array(["Dataset", "IR","i","HS","CS","VM","AMI","ARI"])
		        			writer_details.writerow(header_details)
		        			Flag_details = True
		        		a=np.array([filename, IR, i, float("%0.2f"%(HS[run])), 
		        			float("%0.2f"%(CS[run])),  float("%0.2f"%(VM[run])),  float("%0.2f"%(AMI[run])),  
		        			float("%0.2f"%(ARI[run]))])
		        		writer_details.writerow(a)
		        	out_details.close()

		    if(export_flags['Export_avg']==True):
		    	export_to_file_avg=results_directory + "experiment_avg.csv"
		    	with open(export_to_file_avg, 'a',newline='\n') as out_avg:
		    		writer = csv.writer(out_avg,delimiter=',')
		    		if (Flag_avg==False): # just one time to write the header of the CSV file
			    		header= np.array(["Dataset", "IR","i", "HS","CS","VM","AMI","ARI"])
			    		writer.writerow(header)
			    		Flag_avg=True
			    	avgHS = str(float("%0.2f"%(sum(HS) / NumOfRuns)))
			    	avgCS = str(float("%0.2f"%(sum(CS) / NumOfRuns)))
			    	avgVM = str(float("%0.2f"%(sum(VM) / NumOfRuns)))
			    	avgAMI = str(float("%0.2f"%(sum(AMI) / NumOfRuns)))
			    	avgARI = str(float("%0.2f"%(sum(ARI) / NumOfRuns)))

			    	a=np.array([filename, IR, i, avgHS, avgCS, avgVM, avgAMI, avgARI])
			    	writer.writerow(a)
		    	out_avg.close()

		    if(export_flags['Export_avg']==True):
		    	export_to_file_best=results_directory + "experiment_best.csv"
		    	with open(export_to_file_best, 'a',newline='\n') as out_best:
		    		writer = csv.writer(out_best,delimiter=',')
		    		if (Flag_best==False): # just one time to write the header of the CSV file
			    		header= np.array(["Dataset", "IR","i", "HS","CS","VM","AMI","ARI"])
			    		writer.writerow(header)
			    		Flag_best=True
			    	bestHS = str(float("%0.2f"%(max(HS))))
			    	bestCS = str(float("%0.2f"%(max(CS))))
			    	bestVM = str(float("%0.2f"%(max(VM))))
			    	bestAMI = str(float("%0.2f"%(max(AMI))))
			    	bestARI = str(float("%0.2f"%(max(ARI))))

			    	a=np.array([filename, IR, i, bestHS, bestCS, bestVM, bestAMI, bestARI])
			    	writer.writerow(a)
		    	out_best.close()
		    print("AVG," + avgHS+"," + avgCS +","+ avgVM +"," +avgAMI +"," +avgARI)
		    print("Best," + bestHS+"," + bestCS +","+ bestVM +"," + bestAMI +"," + bestARI) 


if export_flags['Export_boxplot'] == True:
	ev_measures=['HS', 'CS', 'VM', 'AMI', 'ARI']
	box_plot.run(results_directory, indexing_ratio, iterations, filenames, ev_measures)

if export_flags['Export_surface'] == True:
	ev_measures=['HS', 'CS', 'VM', 'AMI', 'ARI']
	#if len(indexing_ratio) < 3 or len(iterations) < 3:
	#	print("The length of indexing_ratio & iterations lists should be at least 3 for the surface to work")
	#else:
	surface.run(results_directory, indexing_ratio, iterations, filenames, ev_measures)

if export_flags['Export_plot_labels'] == True:
	print('To be done')



