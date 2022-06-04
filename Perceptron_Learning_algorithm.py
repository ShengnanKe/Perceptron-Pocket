# -*- coding: utf-8 -*-
"""
Class: DSCI552
Assignment 4
@author: Shengnan Ke
"""

# Implement the Perceptron Learning algorithm. 
# Run it on the data file "classification.txt" ignoring the 5th column. 
# That is, consider only the first 4 columns in each row. 
# The first 3 columns are the coordinates of a point; 
# and the 4th column is its classification label +1 or -1. 
# Report your results (weights and accuracy after the final iteration).

import numpy as np

# loading data
dataset = np.loadtxt("classification.txt",delimiter=",")

# set up learning rate
learning_rate = 0.0001

# set up the dataset:"consider only the first 4 columns in each row"
dataset = np.delete(dataset,[4],axis = 1)
# classification label
labels = dataset[:,-1] # taking all the rows and only the last column

# coordinates of data points
coordinates = np.delete(dataset,[3],axis = 1)

# random generate value for weights
weights = np.random.rand(4,1) # 3 dimension + 1 = 4

placeholder = np.ones((coordinates.shape[0],1)) 
# since we have 4 weight value, we need a additional column set as 1 in the coordinates

coordinates = np.concatenate((placeholder,coordinates),axis = 1)

# initialize for iterations
violated_constraints = 1
iterations = 0

while(violated_constraints != 0):
    iterations += 1
    violated_constraints = 0
    for i in range(0,len(coordinates)):
        current_coordinates = coordinates[i].reshape(-1,coordinates.shape[1])
        current_label = labels[i]
        w_Trans_x = np.dot(current_coordinates, weights)[0][0]
        if current_label == 1 and w_Trans_x<0: # case 1
            violated_constraints += 1
            weights = weights + learning_rate * np.transpose(current_coordinates)
        elif current_label == -1 and w_Trans_x>0: # case 2
            violated_constraints += 1
            weights = weights - learning_rate * np.transpose(current_coordinates)
    if violated_constraints == 0:
        print ("Final Iteration Number:", iterations, "Misclassified Data Points: ",violated_constraints)

print("The accuracy is 100% as default.")
print ("Weights:", weights.transpose())

