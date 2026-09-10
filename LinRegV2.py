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
        Calculates Loss by Sum of Squared Residuals
    '''
    def __init__(self, xAxis, yAxis, epochs = 100, learningRate = 0.001):
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

        self.derivativeGradient = 0
        self.derivativeBias = 0

        self.TrainModel()

    def __str__(self):
        return(f"X Axis: {self.xAxis}\nY Axis: {self.yAxis}\nPredicted Y: {self.predictedY}\nGradient: {self.gradient}\nBias: {self.bias}\nLoss: {self.loss}\ndL/dm: {self.derivativeGradient}\ndL/db: {self.derivativeBias}")


    def TrainModel(self):
        for i in range(self.epochs):
            self.ForwardPropagation()
            self.BackPropagation()
            self.LossFunction()


    def ForwardPropagation(self):
        self.predictedY= np.zeros(self.ARRAY_SIZE)
        for i in range(self.ARRAY_SIZE):
            self.predictedY[i] = self.gradient*self.xAxis[i] + self.bias



    def BackPropagation(self):
        self.derivativeGradient = 0
        for i in range(self.ARRAY_SIZE):
            self.derivativeGradient += 2 * self.xAxis[i] * (self.predictedY[i] - self.yAxis[i])
        # Calculate dL/dm
    
        self.derivativeBias = 0
        for i in range(self.ARRAY_SIZE):
            self.derivativeBias += 2 * (self.predictedY[i] - self.yAxis[i])
        # Calculate dL/db

        self.gradient = self.gradient - self.derivativeGradient * self.learningRate
        # m = m - (dL/dm * α) 
        self.bias = self.bias - self.derivativeBias * self.learningRate
        # b = b - (dL/db * α) 
    

    def LossFunction(self):
        self.loss = 0
        for i in range(self.ARRAY_SIZE):
            self.loss += (self.predictedY[i] - self.yAxis[i])**2

        return self.loss

    
class RegressionGraph():
    ''' 
    Draws a Line of Best Fit on a graph

    Parameters
    ----------
    fig: plt.Figure
    ax: plt.Axes
    model: LinearRegression

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
    xAxis = np.array([i for i in range(0,ARRAY_SIZE)])
    yAxis = np.array([])
    for i in range(ARRAY_SIZE):
        value = round(np.random.uniform(0,10),3)
        yAxis = np.append(yAxis, value)

    # yAxis = np.array([1,4,7,10,13,16,19,22])

    model = Linear_Regression(xAxis, yAxis, epochs=1000, learningRate=0.001)
    print(model)
    RegressionGraph(model, "regression")

    

if __name__=="__main__":
    main()