from imghdr import tests
from DecisionTree import decisionTree
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

def predict(decisionTreeNode, x):
    # print("attribute = ",decisionTreeNode.attribute)
    if decisionTreeNode.isLeaf():
        return decisionTreeNode.label
    x_bar = x[decisionTreeNode.attribute]
    # print(x_bar)
    # print(decisionTreeNode.children)
    if x_bar in decisionTreeNode.children:
        newNode = decisionTreeNode.children[x_bar]
    else:
        print("Fuckkkkkkkkkkk again..!!!!")
        return None
    return predict(newNode,x)

def main():
    dataSet = pd.read_csv("cleanedData.csv")
    # print(dataSet.head())
    # print(dataSet.dtypes)
    trainSet, testSet = trainTestSplit(dataSet=dataSet,testRatio=0.2)
    decisionTreeRoot = decisionTree(trainSet)
    if decisionTreeRoot is None:
        print("fuckkkkkkkk..!!")
        return
    
    accuracy = 0
    # print(testSet)
    for i in range(len(testSet)):
        sample = testSet.iloc[i]
        # print(sample["Profession"])
        y_hat = predict(decisionTreeNode=decisionTreeRoot,x=sample)
        if y_hat==sample["Segmentation"]:
            accuracy+=1
            print("hurrayyy..!! correct accuracy increased",accuracy)
    
    print("Accuracy = ",accuracy/len(testSet))

if __name__ == "__main__":
    main()