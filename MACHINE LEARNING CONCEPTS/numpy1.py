import numpy as np

# Creating a 1D array
arr1 = np.array([1, 2, 3, 4, 5])
print("1D Array:", arr1)

# Creating a 2D array (matrix)
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array:\n", arr2)

# Creating an array of zeros
zeros = np.zeros((3, 3))
print("3x3 Zero Matrix:\n", zeros)

# Creating an array of ones
ones = np.ones((2, 4))
print("2x4 Ones Matrix:\n", ones)

# Creating an array with a range of numbers
range_arr = np.arange(0, 10, 2)  # from 0 to 10 with step 2
print("Range Array:", range_arr)

# Creating an array with equally spaced values
lin_arr = np.linspace(0, 1, 5)  # 5 values between 0 and 1
print("Linearly spaced values:", lin_arr)