'''
Mark III of the Linear Regression Model. 

This version of the Linear Regression model is utilisting the functions of numpy within my code.
Using functions such as np.sum rather than for loops.
Reason: np functions seem to be faster for np.arrays as our dataset gets larger. See testV3.py for more details.
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
        Number of observations. Also called 'N'
        = |xAxis| = |yAxis| = |ŷ|
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
        dL/dm. The derivative of the loss function w.r.t m
    derivativeBias: float
        dL/db. The derivative of the loss function w.r.t b


    Methods
    -------
    __init__(self, xAxis, yAxis, epochs = 100, learningRate = 0.01):
        Initialises an object of the Linear Regression class
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
        # TODO: Need getters and setters for these two
        if np.shape(self.xAxis) != np.shape(self.yAxis):
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
        Trains Model by iterating Forward Propogation, Loss Function and Back Propogation for epoch iterations
        '''
        for i in range(self.epochs):
            self.ForwardPropagation()
            self.BackPropagation()
            self.loss = self.LossFunction()
            # TODO Stopping case once loss increases? But to do this, need data normalisation first.


    def ForwardPropagation(self):
        self.predictedY= self.gradient * self.xAxis + self.bias


    def BackPropagation(self):
        # See README for definitions of dL/dm and dL/db
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
        # Calculate Mean Squared Error
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
    __init__(self, model, mode="regression"):
        Initialises an object of the Linear Regression class
    Draw(self, mode):
        Draws a scatter plot if the mode = "scatter"
        Draws a scatter plot and regression line if mode = "regression" 

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

    ARRAY_SIZE = 100
    xAxis = np.array([i for i in range(ARRAY_SIZE)])
    yAxis = np.array([])
    for i in range(ARRAY_SIZE):
        # value = np.random.random()
        value = np.array(20*xAxis[i]+2)
        yAxis = np.append(yAxis, value)

    model = Linear_Regression(xAxis, yAxis, epochs=10_000, learningRate=0.001)
    model.TrainModel()
    print(model)
    RegressionGraph(model, "regression")

    

if __name__=="__main__":
    main()