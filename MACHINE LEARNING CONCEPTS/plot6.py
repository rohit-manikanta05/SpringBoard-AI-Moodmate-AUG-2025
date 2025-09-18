import matplotlib.pyplot as plt
import numpy as np

data = np.random.randn(1000)  # 1000 random numbers

plt.hist(data, bins=30, color="purple", edgecolor="black")
plt.title("Histogram Example")
plt.show()