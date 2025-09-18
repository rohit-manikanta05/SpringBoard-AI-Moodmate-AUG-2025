import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)    # 100 random numbers between 0 and 2
y = 4 + 3 * X + np.random.randn(100, 1)  # true function + noise

# Step 2: Create and train model
model = LinearRegression()
model.fit(X, y)

# Step 3: Get learned parameters
print("Intercept (b):", model.intercept_)
print("Slope (m):", model.coef_)

# Step 4: Make predictions
X_new = np.array([[0], [2]])    # test inputs
y_predict = model.predict(X_new)

print("Predicted values for x=0 and x=2:")
print(y_predict)

# Step 5: Plot results
plt.scatter(X, y, color="blue", label="Training Data")
plt.plot(X_new, y_predict, color="red", linewidth=2, label="Prediction Line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()