'''
Mark II of the Linear Regression Model. 
Implementing Linear Regression using Object Oriented Programming rather than Procedural.

Why? Because the everything is self contained and no more use of global variables to ensure code safety.
'''

import numpy as np
import matplotlib.pyplot as plt

class Linear_Regression():
    '''
    Finds the line of best fit of a 2D scatter graph of coordinates on the Cartesian plane.

    
    Parameters
    ----------
    xAxis: np.array
    yAxis: np.array 
    ARRAY_SIZE: int
        Number of coordinates in xAxis and yAxis
    predictedY: np.array 
        Also called 'ŷ'
    gradient: float
        Also called 'm'
    bias: float
        Y-intercept parameter. ALso called 'b'
    Loss: float
        The sum of squared residuals calculated from the loss function
    Epochs: int
    learningRate: float 
        Also called 'α'
    derivativeGradient: float
        dL/dm
    derivativeBias: float
        dL/db


    Methods
    -------
    __init__(self):
        Initialises an object of the class
    ForwardPropagation():
        Calculates ŷ = mX + b
    BackPropogation():
        Calculates dL/dm and dL/db and carries out GD
        m = m - dL/dm*α
        b = b - dL/db*α
    LossFunction()
        Calculates L by Sum of Squared Residuals
    '''
    def __init__(self, xAxis, yAxis, epochs = 100, learningRate = 0.01):
        self.xAxis = xAxis
        self.yAxis = yAxis
        # Need getters and setters for these two
        self.ARRAY_SIZE = len(self.xAxis)
        self.predictedY = np.zeros_like(self.xAxis)
        # Figure out how to initilise this. empty array? or filled with zeros
        self.gradient = 0.0
        self.bias = 0.0
        self.loss = 0.0
        self.epochs = epochs
        self.learningRate = learningRate

        self.derivativeGradient = 0
        self.derivativeBias = 0



    def ForwardPropagation(self):
        for i in range(self.ARRAY_SIZE):
            self.predictedY[i] = self.gradient*self.xAxis[i] + self.yIntercept


    def BackPropagation(self):
        self.derivativeGradient = 0
        for i in range(self.ARRAY_SIZE):
            self.derivativeGradient += 2 * self.xAxis[i] * (self.predictedY[i] - self.yAxis[i])
        # Calculate dL/dm
    
        self.derivativeBias = 0
        for i in range(self.ARRAY_SIZE):
            self.derivativeBias += 2 * (self.predictedY[i] - self.yAxis[i])
        # Calculate dL/db
    
        gradient = gradient - self.derivativeGradient * self.learningRate
        # m = m - (dL/dm * α) 
        yIntercept = yIntercept - self.derivativeBias * self.learningRate
        # b = b - (dL/db * α) 
    

    def LossFunction(self):
        self.loss = 0
        for i in range(self.ARRAY_SIZE):
            self.loss += (self.predictedY[i] - self.yAxis[i])**2

        return self.loss

    



def main():
    ARRAY_SIZE = 4
    xAxis = np.array([i for i in range(0,ARRAY_SIZE)])
    yAxis = np.array([])
    np.random.seed(10)
    for i in range(ARRAY_SIZE):
        value = round(np.random.uniform(0,10),3)
        yAxis = np.append(yAxis, value)
    print(yAxis)
    fig, ax = plt.subplots()
    print(type(fig), type(ax))


if __name__=="__main__":
    main()