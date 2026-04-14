import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0, 500.0]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)
print(f"Best Alpha Found: {grid_search.best_params_}") 
print(f"Best R2 Score: {grid_search.best_score_:.4f}") 