from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

# Prediction
print("Prediction for first sample:", clf.predict([X[0]]))

# Visualize tree
plt.figure(figsize=(10,6))
tree.plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()