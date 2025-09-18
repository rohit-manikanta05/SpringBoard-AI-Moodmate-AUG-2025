import numpy as np
from sklearn.preprocessing import StandardScaler

# Example dataset
X = np.array([[10, 100],
              [20, 200],
              [30, 300],
              [40, 400]])

# Standardization (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Original Data:\n", X)
print("Standardized Data:\n", X_scaled)