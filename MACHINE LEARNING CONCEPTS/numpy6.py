import numpy as np
from sklearn.model_selection import train_test_split

# Example dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])  # Features
y = np.array([2, 4, 6, 8, 10, 12, 14, 16])              # Labels

# Splitting the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Data:\n", X_train, y_train)
print("Testing Data:\n", X_test, y_test)