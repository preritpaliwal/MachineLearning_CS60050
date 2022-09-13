from unicodedata import category
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

df = pd.read_csv("Dataset_A.csv")

"""
# Removing NAN with Mode
"""

naList = df.isna().sum()
# print(naList)
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
    col = df1.index[i]
    if df1[i]=="object":
        df[col] = df[col].astype("category").cat.codes
    elif df1[i]=="float64":
        df[col] = df[col].astype("int64")

print(df.head())
df = df.drop(columns=["ID"])
print(df.head())

"""
Grouping Age into 0-9, 10-19, 20-29, etc ranges
"""

df["Age"] = df["Age"]//10

# valCnt = df["Gender"].value_counts()
# dfs = []
# for i in range(len(valCnt)):
#     dfi = df[df["Gender"]==valCnt.index[i]]
#     dfs.append(dfi)
#     print(dfi)


"""
Writing final cleand data to csv file
"""
df.to_csv("cleanedData.csv",index=False)