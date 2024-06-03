
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Sample data
x, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

# K-means clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(x)

#printing cluster centers
kmeans.cluster_centers_

#Plotting the clusters and cluster centers
plt.scatter(x[:, 0], x[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:,1], marker='.', s=200, color='black')
plt.show()