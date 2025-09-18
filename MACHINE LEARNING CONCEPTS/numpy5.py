import numpy as np

# Random integers between 1 and 10 (size=5)
rand_ints = np.random.randint(1, 10, size=5)
print("Random Integers:", rand_ints)

# Random floats between 0 and 1
rand_floats = np.random.rand(5)
print("Random Floats:", rand_floats)

# Random 3x3 matrix
rand_matrix = np.random.randn(3, 3)
print("Random Normal Distribution Matrix:\n", rand_matrix)

# Shuffling
arr = np.array([1, 2, 3, 4, 5])
np.random.shuffle(arr)
print("Shuffled Array:", arr)