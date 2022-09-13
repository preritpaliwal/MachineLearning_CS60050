import pandas as pd
import numpy as np

class Node:
    def __init__(self,attribute):
        self.attribute = attribute
        self.children = {}
        self.label = None
    
    def addChild(self,x):
        self.children[x[0]] = x[1]
    
    def isLeaf(self):
        if self.label is not None:
            return True
        return False

def getEntropy(y):
    total = len(y)
    valCnt = y.value_counts()
    ans = 0
    for i in range(len(valCnt)):
        tmp = valCnt[valCnt.index[i]]/total
        ans -= tmp*np.log(tmp)
    return ans

def getIG(x,y, attr):
    initialEntropy = getEntropy(y=y)
    total = len(x[attr])
    valCnt = x[attr].value_counts()
    finalEntropy = 0
    entropies = []
    for i in range(len(valCnt)):
        yi = y[x[attr]==valCnt.index[i]]
        entropies.append(getEntropy(yi))
    # print(valCnt)
    # print(attr)
    for i in range(len(valCnt)):
        # print("valCnt.index[i] = ",valCnt.index[i])
        # print(valCnt[valCnt.index[i]])
        tmp = valCnt[valCnt.index[i]]/total
        finalEntropy += tmp*entropies[i]
    return initialEntropy - finalEntropy

def getInfoGainList(x,y,attrs):
    infoGain = []
    for attr in attrs:
        # print("calling for attribute : ",attr)
        infoGain.append(getIG(x,y,attr))
    return infoGain

def getNextNode(x,y):
    # All current Attributes
    attributes = x.columns.values
    # print("Attributes : ",attributes)
    if(len(attributes)==0):
        # print("why go here")
        return None

    if(len(attributes)==1):
        valCnt = y.value_counts()
        n = Node(attribute=attributes[0])
        # setting as the one with max frquency
        n.label = valCnt.index[0]
        return n
    
    # find information gain of all attributes 
    # print("finding info gain")
    InfoGainPerAttr = getInfoGainList(x,y,attributes)
    # print(InfoGainPerAttr)
    
    # find attribute with maximum information gain
    i_max = np.argmax(InfoGainPerAttr)
    
    if InfoGainPerAttr[i_max]<0.0001:
        valCnt = y.value_counts()
        n = Node(attribute=attributes[i_max])
        # setting as the one with max frquency
        n.label = valCnt.index[0]
        return n
    
    # create a node with that as attribute
    n = Node(attribute=attributes[i_max])
    # print("Node attribute = ",n.attribute)
    # find all dataSet seperated based on attribute with max information gain
    valCnt = x[attributes[i_max]].value_counts()
    for i in range(len(valCnt)):
        xi = x[x[attributes[i_max]]==valCnt.index[i]]
        yi = y[x[attributes[i_max]]==valCnt.index[i]]
        xi = xi.drop(columns=attributes[i_max])
        child = getNextNode(x=xi,y=yi)
        if child is not None:
            n.addChild((valCnt.index[i],child))
    return n

def decisionTree(TrainData):
    x = TrainData.drop(columns="Segmentation")
    y = TrainData["Segmentation"]
    # print(x.head())
    # print(y.head())
    return getNextNode(x,y)

def predict(decisionTreeNode, x):
    if decisionTreeNode.isLeaf():
        return decisionTreeNode.label
    x_bar = x[decisionTreeNode.attribute]
    if x_bar in decisionTreeNode.children:
        newNode = decisionTreeNode.children[x_bar]
    else:
        # print("Fuckkkkkkkkkkk again..!!!!")
        return None
    return predict(newNode,x)

def getAccuracy(root, testSet):
    accuracy = 0
    for i in range(len(testSet)):
        sample = testSet.iloc[i]
        y_hat = predict(decisionTreeNode=root,x=sample)
        if y_hat==sample["Segmentation"]:
            accuracy+=1
            # print("hurrayyy..!! correct accuracy increased",accuracy)
    accuracy = accuracy/len(testSet)
    print("Accuracy = ",accuracy)
    return accuracy

def pruneNode(root, curNode, validationSet,depth=0):
    if curNode.isLeaf():
        return
    i = 0
    for child in curNode.children:
        print("before i = ",i," at depth : ",depth)
        pruneNode(root=root,curNode=curNode.children[child],validationSet=validationSet,depth=depth+1)
        print("done with child i = ",i," at depth : ",depth)
        i += 1
    
    initialAccuracy = getAccuracy(root, validationSet)
    valCnt = validationSet["Segmentation"].value_counts()
    curNode.label = valCnt.index[0]
    newAccuracy = getAccuracy(root, validationSet)
    if newAccuracy < initialAccuracy + 0.001:
        print("\n\n\nundoo pruning.........\n\n\n")
        curNode.label = None
    return