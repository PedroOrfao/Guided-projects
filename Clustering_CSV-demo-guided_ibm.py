import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Sample url provided
URL =  "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/customers.csv"

#loading the data into a dataframe
df = pd.read_csv(URL)
# amount of clusters
number_of_clusters = 3
cluster = KMeans(n_clusters = number_of_clusters)

# Training the model on dataset
result = cluster.fit(df[['Fresh_Food', 'Milk']])

plt.scatter(df['Fresh_Food'], df['Milk'], c=cluster.labels_, cmap='viridis')
plt.scatter(cluster.cluster_centers_[:, 0],cluster.cluster_centers_[:, 1], marker='.', s=200, color='black')
plt.xlabel('Fresh_Food')
plt.ylabel('Milk')
plt.title('Cluster of costumers')
plt.show()
