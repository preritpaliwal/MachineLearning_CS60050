from DecisionTree import decisionTree,pruneNode,getAccuracy
import pandas as pd
import numpy as np


"""
data splitting
"""
def trainTestSplit(dataSet, testRatio,shuffle=True):
    if shuffle:
        dataSet = dataSet.sample(frac=1)
    n = len(dataSet)
    n_test = int(n*(testRatio))
    testSet = dataSet[:n_test]
    trainSet = dataSet[n_test:]
    return trainSet,testSet

def main():
    dataSet = pd.read_csv("cleanedData.csv")
    # print(dataSet.head())
    # print(dataSet.dtypes)
    trainSet, testSet = trainTestSplit(dataSet=dataSet,testRatio=0.2)
    trainSet, validationSet = trainTestSplit(dataSet=trainSet, testRatio=0.1)
    decisionTreeRoot = decisionTree(trainSet)
    if decisionTreeRoot is None:
        print("fuckkkkkkkk..!!")
        return
    accuracy = getAccuracy(decisionTreeRoot,testSet=testSet)
    print("Before Pruning Accuracy: ", accuracy)
    pruneNode(decisionTreeRoot,decisionTreeRoot,validationSet=validationSet)
    accuracy = getAccuracy(decisionTreeRoot,testSet)
    print("After Pruning Accuracy: ", accuracy)



if __name__ == "__main__":
    main()