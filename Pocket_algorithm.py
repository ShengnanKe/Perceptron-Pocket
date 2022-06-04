# -*- coding: utf-8 -*-
"""
Class: DSCI552
Assignment 4
@author: Shengnan Ke
"""

# Implement the Pocket algorithm and run it on the data file "classification.txt" 
# ignoring the 4th column. That is, consider only the first 3 columns and the 5th column in each row.
# The first 3 columns are the coordinates of a point; 
# and the 5th column is its classification label +1 or -1. 
# Plot the number of misclassified points against the number of iterations of the algorithm. 
# Run up to 7000 iterations. 
# Also, report your results (weights and accuracy after the final iteration).

import numpy as np
import matplotlib.pyplot as plt

# loading data
dataset = np.loadtxt("classification.txt",delimiter=",")

# set up learning rate
learning_rate = 0.001

# "consider only the first 3 columns and the 5th column in each row"
dataset = np.delete(dataset,[3],axis = 1)
# classification label
labels = dataset[:,-1] # taking all the rows and only the last column

# coordinates of data points
coordinates = np.delete(dataset,[3],axis = 1)

# random generate value for weights
weights = np.random.rand(4,1) # 3 dimension + 1 = 4

placeholder = np.ones((coordinates.shape[0],1)) 
# since we have 4 weight value, we need a additional column set as 1 in the coordinates

coordinates = np.concatenate((placeholder,coordinates),axis = 1)

# create a new list to save the data for plot(misclassified points against the number of iterations)
plotData = []

# initialize for iterations
violated_constraints = 1
iterations = 0
min_violated_constraints = 2000  # here are 2000 data in total

while (violated_constraints != 0 and (iterations < 7000)):
    iterations += 1
    violated_constraints = 0
    for i in range(0, len(coordinates)):
        current_coordinates = coordinates[i].reshape(-1,coordinates.shape[1])
        current_label = labels[i]
        w_Trans_x = np.dot(current_coordinates, weights)[0][0]
        if current_label == 1 and w_Trans_x<0: # case 1
            violated_constraints += 1
            weights = weights + learning_rate * np.transpose(current_coordinates)
        elif current_label == -1 and w_Trans_x>0: # case 2
            violated_constraints += 1
            weights = weights - learning_rate * np.transpose(current_coordinates)
    # append the number of missed classifications into the ploting data list
    plotData.append(violated_constraints)
    if violated_constraints < min_violated_constraints:
        min_violated_constraints = violated_constraints

print ("Minimum Misclassifications: ", min_violated_constraints)
print("Final weights: ", weights.transpose())
print ("Accuracy: ",(((2000 - min_violated_constraints)/2000)*100),"%")

plt.plot(np.arange(0,7000),plotData,'g')
plt.title(" Number of Misclassified Points V.S. Number of Iterations ")
plt.xlabel("Number of Iterations")
plt.ylabel("Number of Misclassified Points")
plt.show()
