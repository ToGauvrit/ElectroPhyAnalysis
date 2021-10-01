
ElectroPhyAnalysis (EPA)
========================
Welcome to the documentation about EPA

The Project
-----------
The project is to make a bundle of tools to analyse single cell Axon Binary Format (abf) recordings. There is for the moment only spontaneous recordings that are considered. Only group1 vs group2 comparaison analysis can be made.  

API
-----------
* PSDAnalysis.py : plot welch periodogram and create excel file with the absolute power spectral density per frequency bands for the two groups of files

* UpstateDownstateAnalysis.py : compute Upstate/Downstate detection and measurements of duration, frequency and Vm. This method uses medians of the signal to discrimninate between the states. The process is slow. Write two excel files, one that describe every states for every cells and the second that show the measurements by cell and group.


* UpstateDownstateAlendaMethod.py : Inspired initially by Alenda et al (2010) paper this method differs from it now on with some changes. The threshold are defined by slopes in the density histogram of the signal. It return a excel filethat show the measurements by cell and group.
