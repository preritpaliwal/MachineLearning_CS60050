import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Kmeans import getInitialRep,Kmeans

df = pd.read_csv("wine.csv")

x = df.drop(columns=["Target Class"],axis=1)
y = df["Target Class"]


x_normalised = StandardScaler().fit_transform(x)


pca = PCA(0.95)
principalComponents = pca.fit_transform(x_normalised)
principalDf = pd.DataFrame(data = principalComponents)

print(pca.explained_variance_ratio_)
plt.plot(pca.explained_variance_ratio_)
plt.savefig("Variance_VS_Component_number.png", format="png")
plt.show()

K,NMI = [],[]
for k in range(2,9):
    random = False
    J, reps, nmi = Kmeans(getInitialRep(k,principalDf,random),principalDf,y)
    K.append(k)
    NMI.append(nmi)

print(K)
print(NMI)
plt.plot(K,NMI)
plt.savefig("K_VS_NMI.png", format="png")
plt.show()