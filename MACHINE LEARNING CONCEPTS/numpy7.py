import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
X = np.array([[1], [2], [3], [4], [5]])  # Feature
y = np.array([2, 4, 6, 8, 10])           # Label (y = 2x)

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
pred = model.predict([[6]])
print("Prediction for 6:", pred)

# Model parameters
print("Slope (Coefficient):", model.coef_)
print("Intercept:", model.intercept_)