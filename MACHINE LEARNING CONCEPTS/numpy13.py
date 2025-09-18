from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt

# Sample dataset
X = np.array([[1, 2], [2, 3], [3, 4],
              [8, 7], [9, 6], [10, 8]])

# Hierarchical Clustering (2 clusters)
clustering = AgglomerativeClustering(n_clusters=2)
labels = clustering.fit_predict(X)

print("Cluster Labels:", labels)

# Visualization
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.title("Agglomerative (Hierarchical) Clustering")
plt.show()