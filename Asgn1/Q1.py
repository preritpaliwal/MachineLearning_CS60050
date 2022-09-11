from re import I
import pandas as pd
import math

class Node:
    def __init__(self,attribute):
        self.attribute = attribute
        self.children = []
    
    def addChild(self,x):
        self.children.append(x)

def getDFs(df, attr):
    dfs = []
    valCnt = df[attr].value_counts()
    for i in range(len(valCnt)):
        dfi = df[df[attr]==valCnt.index[i]]
        dfs.append(dfi)
    return dfs

def getEntropy(df):
    total = len(df["Segmentation"])
    valCnt = df["Segmentation"].value_counts()
    ans = 0
    for i in range(len(valCnt)):
        tmp = valCnt[i]/total
        ans -= tmp*math.log(tmp)
    return ans

def getIG(df, attr):
    initialEntropy = getEntropy(df)
    total = len(df[attr])
    valCnt = df[attr].value_counts()
    finalEntropy = 0
    dfs = getDFs(df,attr)
    print(valCnt)
    for i in range(len(dfs)):
        print(valCnt[i])
        tmp = valCnt[i]/total
        finalEntropy += tmp*getEntropy(dfs[i])
    return initialEntropy - finalEntropy

def getInfoGainList(df,attrs):
    infoGain = []
    for attr in attrs:
        infoGain.append(getIG(df,attr))
    return infoGain

def getNextNode(df):
    attributes = df.columns.values
    
    InfoGainPerAttr = getInfoGainList(df,attributes)
    
    i_max = 0
    for i in range(1,len(attributes)):
        if InfoGainPerAttr[i]>InfoGainPerAttr[i_max]:
            i_max = i
    
    n = Node(attribute=attributes[i_max])
    dfs = getDFs(df,attributes[i_max])
    for df in dfs:
        n.addChild(getNextNode(df=df))
    return n

df = pd.read_csv("cleanedData.csv")
# print(df.head())

decisionTreeClassifier = getNextNode(df)