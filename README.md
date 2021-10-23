# CSE572-Assignment2   
#Name:Wenzhe Zheng   
#Date:10/23/2021
#Before running project, please read through this instructuction:
Contained within the folder is train.py and  test.py
the train.py program reads four files;the train.py reads CGMData.csv, CGM_patient2.csv and InsulinData.csv, Insulin_patient2.csv, 
The train.py  extracts meal and no-meal data, extracts features, trains your machine to recognize meal and no-meal classes, and stores the machine in a pickle file (Python API pickle).
The test.py reads test.csv which has the N x 24 matrix and outputs a Result.csv file which has N x 1 vector of 1s and 0s, where 1 denotes meal, 0 denotes no meal.
Assume that CGMData.csv, CGM_patient2.csv and InsulinData.csv, Insulin_patient2.csv files 
the test.py program generates result.csv file with 1 for meal and 0 for no meal
pandas module was used to extract and manipulate data
while sklearn package provided modules and functions for 
processing and provide methods for creating machine
the SVC was used to create training model
