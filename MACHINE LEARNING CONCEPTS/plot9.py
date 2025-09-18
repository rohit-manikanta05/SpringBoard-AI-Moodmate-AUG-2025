import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 5, 7, 1, 6]

plt.plot(x, y, color="blue", linestyle="--", marker="o", linewidth=2, markersize=8)
plt.grid(True)
plt.title("Customized Plot")
plt.show()