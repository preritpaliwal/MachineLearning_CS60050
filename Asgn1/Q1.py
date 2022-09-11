from unicodedata import category
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

df = pd.read_csv("Dataset_A.csv")

"""
# Removing NAN with Mode
"""

naList = df.isna().sum()
for i in range(len(naList)):
    if(naList[i]>0):
        col = naList.index[i]
        valCnt = df[col].value_counts()
        df = df.fillna({col:valCnt.index[0]})

# print(df.isna().sum())

"""
LABEL ENCODING
"""
df1 = df.dtypes
for i in range(len(df1)):
    if df1[i]=="object":
        col = df1.index[i]
        df[col] = df[col].astype("category").cat.codes

print(df.head())
df = df.drop(columns=["ID"])
print(df.head())
df.to_csv("cleanedData.csv",index=False)