import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = 6*np.random.rand(100,1)-3
y = 0.5*X**2 + X + 2 + np.random.randn(100,1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

X_new = np.linspace(-3, 3, 100).reshape(100, 1) 
X_new_poly = poly_features.transform(X_new) 
y_new = model.predict(X_new_poly) 
plt.scatter(X, y, color='blue', label='Data') 
plt.plot(X_new, y_new, color='red', label='Polynomial Prediction') 
plt.legend() 
plt.show() 


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=3)

tree_reg.fit(X, y)

y_tree_pred = tree_reg.predict(X_new)

plt.plot(X_new, y_tree_pred, color='green', label='Decision Tree')
plt.legend()
plt.show()