'''
This is the first iteration of Linear Regression
It is a simple model which uses procedural programming in order to calculate the line of best fit.
The data is randomly generated and is not normalised. Standard gradient descent is used.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def main():
    # Initialise Constants
    global ARRAY_SIZE
    ARRAY_SIZE = 10
    global EPOCHS
    EPOCHS = 1000
    global LEARNING_RATE
    LEARNING_RATE = 0.001

    # Initialise parameters as random
    gradient = random.random()
    yIntercept = random.random()

    # Initialise X Axis as array of integers and Y Axis as empty array
    xAxis = np.array([i for i in range(0,ARRAY_SIZE)])
    yAxis = np.array([])

    for i in range(ARRAY_SIZE):
        value = round(random.uniform(0,10),3)
        yAxis = np.append(yAxis, value)
        # Create an Array of size: ARRAY_SIZE with floats from 0 to 10. 
    
    
    # Initialise and set up figure/axes 
    fig, (ax1, ax2)= plt.subplots(1,2)
    fig.set_figwidth(11)
    fig.set_figheight(5)

    # Plot initial data points on both axes
    ax1.plot(xAxis, yAxis, "o", color="b")
    predictedY = ForwardPropogation(xAxis, gradient, yIntercept)
    sumOfSquaredResiduals = ErrorFunction(yAxis, predictedY)
    print(f"Initial Loss: {sumOfSquaredResiduals}")
    ax1.plot(xAxis, predictedY, color= "orange")

    ax2.plot(xAxis, yAxis, "o", color = "b")

    # Let the training begin!
    predictedY, gradient, yIntercept = TrainModel(xAxis, yAxis, gradient, yIntercept)
    sumOfSquaredResiduals = ErrorFunction(yAxis, predictedY)
    print(f"Final Loss: {sumOfSquaredResiduals}")

    # Plot Line of Best Fit
    ax2.plot(xAxis, predictedY, color="orange")
    
    
    plt.show()   

def TrainModel(xAxis, yAxis, gradient, yIntercept):
    for i in range(EPOCHS):
        predictedY = ForwardPropogation(xAxis, gradient, yIntercept)
        gradient, yIntercept = BackPropogation(xAxis, yAxis, predictedY, gradient, yIntercept)
       
    return predictedY, gradient, yIntercept



def ForwardPropogation(xAxis, gradient, yIntercept):
    predictedY = np.zeros(ARRAY_SIZE)
    for i in range(ARRAY_SIZE):
        predictedY[i] = gradient*xAxis[i] + yIntercept

    return predictedY

def BackPropogation(xAxis, yAxis, predictedY, gradient, yIntercept):
    derivativeGradient = 0
    for i in range(ARRAY_SIZE):
        derivativeGradient += 2 * xAxis[i] * (predictedY[i] - yAxis[i])
    # Calculate dL/dm

    derivativeYIntercept = 0
    for i in range(ARRAY_SIZE):
        derivativeYIntercept += 2 * (predictedY[i] - yAxis[i])
    # Calculate dL/db

    gradient = gradient - derivativeGradient * LEARNING_RATE
    # m = m - (dL/dm * α) 
    yIntercept = yIntercept - derivativeYIntercept * LEARNING_RATE
    # b = b - (dL/db * α) 

    return gradient, yIntercept

    

def ErrorFunction(yAxis, predictedY):
    sum = 0
    for i in range(ARRAY_SIZE):
        sum += (predictedY[i] - yAxis[i])**2

    return sum

if __name__=="__main__":
    main()