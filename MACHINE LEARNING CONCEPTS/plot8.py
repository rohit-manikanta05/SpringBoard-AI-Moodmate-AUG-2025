import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [i**2 for i in x]

plt.subplot(1, 2, 1)  # 1 row, 2 cols, 1st plot
plt.plot(x, y, "r-")
plt.title("Line Plot")

plt.subplot(1, 2, 2)  # 1 row, 2 cols, 2nd plot
plt.bar(x, y)
plt.title("Bar Plot")

plt.tight_layout()
plt.show()