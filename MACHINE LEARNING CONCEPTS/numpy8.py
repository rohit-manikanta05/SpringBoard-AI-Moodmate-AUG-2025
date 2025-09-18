import numpy as np
from sklearn.linear_model import LogisticRegression

# Simple dataset: study hours → pass/fail
X = np.array([[1], [2], [3], [4], [5], [6], [7]])
y = np.array([0, 0, 0, 1, 1, 1, 1])  # 0 = Fail, 1 = Pass

# Model
model = LogisticRegression()
model.fit(X, y)

# Predictions
print("Prediction for 2.5 hours:", model.predict([[2.5]]))
print("Prediction for 6 hours:", model.predict([[6]]))

# Probabilities
print("Probabilities for 2.5 hours:", model.predict_proba([[2.5]]))