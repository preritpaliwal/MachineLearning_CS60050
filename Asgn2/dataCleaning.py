import pandas as pd

df = pd.read_csv("wine.data")
cols=["Target Class","Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", 
        "Color intensity","Hue","OD280/OD315 of diluted wines", "Proline"]
df.columns = cols
print(df)
df.to_csv("wine.csv",index=False)