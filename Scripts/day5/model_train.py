from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd

data = {'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
'Score': [35, 40, 55, 60, 68, 72, 81, 88, 92, 95]}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training items: {len(X_train)} | Testing items: {len(X_test)}")


model = LinearRegression()

model.fit(X_train, y_train)

predictions   = model.predict(X_test)

print("Predictions:", predictions)
print("Actual:", y_test.values)


mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MSE:", mse)
print("R2 Score:", r2)