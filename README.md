### This project contains the code used in the analysis,research and visualization of the Thesis.

There are two main directories : BayesianNetwork and OnPLS


- ####OnPLS: This directory contains the notebook 'ExploringTheData'.The notebook contains some data cleaning, analysis and feature selection using OnPLS.
    This notebook does not need to be run in order to run the other files, since the files that were written in that notebook are already present in 'BayesianNetwork/Data'.
    The computation of the files in 'OnPLS/Data' is present in 'BayesianNetwork/BN_Notebooks/FeatureSelection.ipynb'.
    Important Note: The code in 'OnPLS/OnPLS' is take from github repository of the creator of the OnPLS method - https://github.com/tomlof/OnPLS/tree/master

- ####BayesianNetwork: This directory contains all the Pipelines and Significance tests performed wrt Bayesian Networks.
  - In 'BayesianNetwork/Pipelines' are the methods used to compute the different configurations of the Bayesian Network (BayesPipeline.py and BayesRealEdgesPipeline.py), as well the Random Forest (RFPipeline.py).
  - In 'BayesianNetwork/BN_Notebooks'  are notebooks used to compute the significance tests presented in the thesis, as well as a FeatureSelection notebook that aggregated the data for Correlation Selection Method.
  - The files from the 'Data' folder were mostly computed in the 'ExploringTheData' and 'FeatureSelection' notebooks, in addition to the files containing the original data.
  - The files from 'Logs' are computed in BayesPipeline.py, they contain the raw results from the experiments described in the paper.

    
Note: The results from BayesRealEdgesPipeline.py were not used in the Thesis at all, hence why the file is not refactored and may throw some errors, it was left as it contains an interesting approach, performing inference using real edges.