import math
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Binary SVM Class
class BinarySVM:
    def __init__(self, xTrain, yTrain, kernel) -> None:
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.kernel = kernel
        self.yTest = None
        self.__model = None
    
    # Main function for training against instances
    def fitModel(self, degree=2):
        if(self.kernel == 'poly'):
            self.__model = SVC(kernel=self.kernel, gamma=1, degree=degree)
        else:
            self.__model = SVC(kernel=self.kernel, gamma=1)

        self.__model.fit(self.xTrain, self.yTrain)
    
    # Function for predicting from a given test set
    def getPredict(self, xTest):
        if(self.__model == None): 
            print("Train the model first")
            return

        self.yTest = self.__model.predict(xTest)
        return self.yTest

    # Function for returning accuracy of the model
    def accuracy(self, yTestTrue):
        if(self.__model == None): 
            print("Train the model first")
            return

        return np.mean(yTestTrue == self.yTest)

    # Function for returning error of the model
    def error(self, yTestTrue):
        return np.mean(yTestTrue != self.yTest)

# MultiLayer Perceptor Class
class MLP:
    def __init__(self, hidden_layer_sizes, xTrain, yTrain, learning_rate=1e-3, solver='sgd', batch_size=32) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.yTest = None
        self.learning_rate = learning_rate
        self.solver = solver
        self.batch_size = batch_size
        self.__model = None

    # Main function for training against instances
    def fitModel(self):
        self.__model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, learning_rate_init=self.learning_rate, solver=self.solver, batch_size=self.batch_size, max_iter=2000, random_state=1)
        self.__model.fit(self.xTrain, self.yTrain)

    # Function for predicting from a given test set
    def getPredict(self, xTest):
        if(self.__model == None): 
            print("Train the model first")
            return

        self.yTest = self.__model.predict(xTest)
        return self.yTest

    # Function for returning accuracy of the model
    def accuracy(self, yTestTrue):
        if(self.__model == None): 
            print("Train the model first")
            return

        return np.mean(yTestTrue == self.yTest)

    # Function for returning error of the model
    def error(self, yTestTrue):
        return np.mean(yTestTrue != self.yTest)