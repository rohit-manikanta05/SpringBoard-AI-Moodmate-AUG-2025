import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60])

# Accessing elements
print("First element:", arr[0])
print("Last element:", arr[-1])

# Slicing
print("First 3 elements:", arr[0:3])
print("Every second element:", arr[::2])

# Modifying elements
arr[2] = 99
print("Modified array:", arr)

# 2D array slicing
mat = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print("Element at (1,2):", mat[1, 2])   # row 1, col 2
print("First row:", mat[0, :])
print("Second column:", mat[:, 1])