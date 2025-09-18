import matplotlib.pyplot as plt

sizes = [20, 30, 25, 25]
labels = ["Apples", "Bananas", "Cherries", "Dates"]

plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
plt.title("Pie Chart Example")
plt.show()