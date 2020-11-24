import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MiniBatchKMeans

data = pd.read_pickle('../pickle/trainingset.pkl')
independent_cols = data.iloc[:, 0:-1]
print(independent_cols.shape)
wcss = []
for i in range(5000, 5001, 100):
    kmeans = MiniBatchKMeans(n_clusters=i, init='k-means++', n_init=5, max_iter=200, random_state=None, batch_size=50000, verbose=1)
    kmeans.fit(independent_cols.values)
    print(kmeans.inertia_)
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)
    wcss.append(kmeans.inertia_)

# plt.plot(range(100, 1001, 100), wcss)
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()