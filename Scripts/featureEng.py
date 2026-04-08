from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = np.array([[22, 50000], [45, 120000], [30, 80000]])

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

print("Original:\n", data)
print("\nScaled:\n", scaled_data)