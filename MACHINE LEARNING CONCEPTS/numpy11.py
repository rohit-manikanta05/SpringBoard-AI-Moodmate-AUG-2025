import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample dataset (2D points)
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

# Create KMeans model (2 clusters)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Cluster centers and labels
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Labels:", kmeans.labels_)

# Visualization
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            color='red', marker='X', s=200, label="Centroids")
plt.legend()
plt.title("KMeans Clustering")
plt.show()