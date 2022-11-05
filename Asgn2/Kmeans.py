import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
"""
Function to get initial cluster representatives.
"""
def getInitialRep(k,data,random):
    rep = []
    for i in range(k):
        if random:
            rep.append(np.random.rand(28,28)*255)
        else:
            rep.append(data.iloc[i])
    return rep


"""
Function for K-Means Clustering which takes in initial representatives and returns list of Jclust value over the iterations
It converges when assignment of none of the image changes i.e. in other words Jclust does not decrease and remains constant instead.
"""
def Kmeans(rep,data,y):
    N = len(data)
    maxIter = 100
    replabel = [-1]*N
    Jclust = []
    for i in range(maxIter):
        allsame = True
        # print(i)
        loss = 0
        # clustering based on current reps
        for j in range(N):
            dist = []
            for q in range(len(rep)):
                dist.append(np.linalg.norm(data.iloc[j]-rep[q]))
            if(np.argmin(dist)!=replabel[j]):
                replabel[j] = np.argmin(dist)
                allsame = False
            loss += np.min(dist)
        Jclust.append(loss/N)
        if allsame:
            NMI = normalized_mutual_info_score(replabel,y)
            print(f"For k = {len(rep)}, terminating at {i-1}th iteration because no change in assignment, thus algo has converged at loss = {loss/N}, NMI = {NMI}")
            break
            
        # updating reps based on new clustering
        seen = np.zeros((len(rep),1))
        for j in range(N):
            rep[replabel[j]] = (rep[replabel[j]]*seen[replabel[j]] + data.iloc[j])/(seen[replabel[j]]+1)
            seen[replabel[j]] += 1;
    return Jclust, rep, NMI