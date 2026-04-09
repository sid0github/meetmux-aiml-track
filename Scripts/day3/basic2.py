import numpy as np

# dataPoints = np.array([10, 20, 30, 40])

# matrix = dataPoints.reshape(2, 2)
# print(matrix)

# processed_data = dataPoints * 2 #NumPy uses vectorized operations, which are faster because they run at a lower (C-level / hardware level) instead of Python loops.
# print(processed_data)

data = np.array([1, 2, 3, 4, 5, 6])
matrix = data.reshape(2, 3)
print(matrix.shape)