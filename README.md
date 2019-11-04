# NPIR: An efficient clustering algorithm based on the k-nearest neighbors with an indexing ratio
Nearest Point with Indexing Ratio (NPIR) is a clustering algorithm which explores the characteristics of the data points to group similar data points into the same clusters and dissimilar data points into different clusters. It is based on the nearest neighbor search technique in finding a k-nearest neighbor to a certain point. The algorithm iterates to assign data points to the most suitable clusters. It performs Election, Selection, and Assignment operations to assign data points to appropriate clusters.

## Installation
- Python 3.xx is required.

Run

    pip3 install -r requirements.txt

(possibly with `sudo`)

That command above will install  `sklearn`, `NumPy`, `SciPy`, and `treelib` for
you.

- If you are installing NPIR algorithm onto Windows, please Install Anaconda from here https://www.continuum.io/downloads, which is the leading open data science platform powered by Python.

## Get the source

Clone the Git repository from GitHub

    git clone https://github.com/RaneemQaddoura/NPIR


## Quick User Guide

NPIR contains the main file is the NPIR.py, which represents the implementation of the algorithm. The example.py is an example file for using the NPIR algorithm as an interface to the algorithm. In the example.py you can setup your experiment by selecting the datasets, IR parameter, number of iterations, and number of runs. 

The following is a sample example to use the NPIR algorithm.  

Change dataset names, IR parameter, number of iterations, and number of runs variables as you want:  
```
filenames = ["VaryDensity.csv","jain.csv"]
IR_ = [0.15, 0.15]#IR: The indexing ratio to be used for generating the maximum index
i_ = [30, 50]#The number of iteration i
runs = 30
```

Now your experiment is ready to run. Enjoy!


