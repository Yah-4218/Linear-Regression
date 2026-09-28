'''
Mark III of the Linear Regression Model. 

This version of the Linear Regression model is utilisting the functions of numpy within my code.
Using functions such as np.sum rather than for loops.
Reason: np functions seem to be faster for np.arrays. See testV3.py for more details.
'''

import numpy as np
import matplotlib.pyplot as plt
import time

class Linear_Regression():
    '''
    Finds the line of best fit of a 2D scatter graph of coordinates on the Cartesian plane.

    Parameters
    ----------
    xAxis: np.array
    yAxis: np.array 
    ARRAY_SIZE: int
        Number of observations. Also called 'N'
    predictedY: np.array 
        Also called 'ŷ'
    gradient: float
        Also called 'm'
    bias: float
        Y-intercept parameter. Also called 'b'
    Loss: float
        The mean squared error calculated from the loss function
    Epochs: int
    learningRate: float 
        Also called 'α'
    derivativeGradient: float
        dL/dm
    derivativeBias: float
        dL/db


    Methods
    -------
    __init__(self, xAxis, yAxis, epochs = 100, learningRate = 0.01):
        Initialises an object of the class
    __str__(self):
        Returns the instance variables of the object
    ForwardPropagation(self):
        Calculates ŷ = mX + b
    BackPropogation(self):
        Calculates dL/dm and dL/db and carries out GD
        m = m - dL/dm*α
        b = b - dL/db*α
    LossFunction(self)
        Calculates Loss by Mean Squared Error
    '''
    def __init__(self, xAxis, yAxis, epochs = 1_000, learningRate = 0.001):
        self.xAxis = xAxis
        self.yAxis = yAxis
        # Need getters and setters for these two
        if np.shape(self.xAxis)!= np.shape(self.yAxis):
            raise ValueError("X Axis and Y Axis arrays should be the same shape")
        self.ARRAY_SIZE = len(self.xAxis)
        self.predictedY = np.zeros_like(self.xAxis)
        self.gradient = 0.0
        self.bias = 0.0
        self.loss = 0.0
        self.epochs = epochs
        self.learningRate = learningRate

        self.derivativeGradient = 0.0
        self.derivativeBias = 0.0

    def __str__(self):
        return(f"X Axis: {self.xAxis}\nY Axis: {self.yAxis}\nPredicted Y: {self.predictedY}\nGradient: {self.gradient}\nBias: {self.bias}\nLoss: {self.loss}\ndL/dm: {self.derivativeGradient}\ndL/db: {self.derivativeBias}")


    def TrainModel(self):
        '''
        Trains Model by iterating Forward Propogation, Loss Function and Back Propogation for epoch iterations until Loss increases
        '''
        for i in range(self.epochs):
            self.ForwardPropagation()
            newLoss = self.LossFunction()
            self.BackPropagation()

            if i == 0:
                self.loss = newLoss
                continue
            
            oldLoss = self.loss
            if newLoss > oldLoss:
                print(f"Training finished on Epoch {i}")
                print(f"newLoss: {newLoss}, oldLoss: {oldLoss}")
                break; 
            self.loss = newLoss
            

    def ForwardPropagation(self):
        self.predictedY= self.gradient * self.xAxis + self.bias


    def BackPropagation(self):
        self.derivativeGradient = np.sum(2*self.xAxis * (self.predictedY - self.yAxis))
        self.derivativeGradient = self.derivativeGradient/self.ARRAY_SIZE
        
        # Calculate (dL/dm)/N
    
        self.derivativeBias = np.sum(2*(self.predictedY - self.yAxis))
        self.derivativeBias = self.derivativeBias/self.ARRAY_SIZE

        # Calculate (dL/db)/N

        self.gradient = self.gradient - self.derivativeGradient * self.learningRate
        # m = m - (dL/dm * α) 
        self.bias = self.bias - self.derivativeBias * self.learningRate
        # b = b - (dL/db * α) 
    

    def LossFunction(self):
        loss = np.sum((self.predictedY - self.yAxis)**2)
        loss = loss/self.ARRAY_SIZE
        # Mean Squared Error
        return loss

    
class RegressionGraph():
    ''' 
    Draws a Line of Best Fit on a graph

    Parameters
    ----------
    fig: plt.Figure
    ax: plt.Axes
    model: LinearRegression
    mode: ...

    Methods
    -------
    __init__(self, model, mode="regression")

    '''
    def __init__(self, model, mode="regression"):
        self.fig, self.ax = plt.subplots()
        self.model = model
        self.mode = mode

        self.Draw(mode = self.mode)

    def Draw(self, mode="regression"):
        if mode =="regression":
            self.ax.plot(self.model.xAxis, self.model.yAxis, "o", color="blue")
            self.ax.plot(self.model.xAxis, self.model.predictedY, color="orange")

        if mode=="scatter":
            self.ax.plot(self.model.xAxis, self.model.yAxis, "o", color="blue")

        plt.show()



def main():

    ARRAY_SIZE = 8
    xAxis = np.array([i for i in range(ARRAY_SIZE)])
    yAxis = np.array([])
    for i in range(ARRAY_SIZE):
        value = np.random.random()
        yAxis = np.append(yAxis, value)

    model = Linear_Regression(xAxis, yAxis, epochs=10_000, learningRate=0.001)
    model.TrainModel()
    print(model)
    RegressionGraph(model, "regression")

    

if __name__=="__main__":
    main()