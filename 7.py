from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("7.csv")
x1=data['x'].values
x2=data['y'].values
print(data)
X=np.matrix(list(zip(x1,x2)))
plt.scatter(x1,x2)
plt.show()
markers=['s','o','v']
k=3
clusters=KMeans(n_clusters=k).fit(X)
for i,l in enumerate(clusters.labels_):
 plt.plot(x1[i],x2[i],marker=markers[l])
