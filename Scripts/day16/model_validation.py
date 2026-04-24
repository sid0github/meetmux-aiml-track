import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

digits = load_digits()
X, y = digits.data, digits.target

model = RandomForestClassifier(n_estimators=50)
scores = cross_val_score(model,X,y,cv=5)

print(f"Scores for each fold: {scores}") 
print(f"Mean Accuracy: {np.mean(scores):.4f}") 
print(f"Standard Deviation (Stability): {np.std(scores):.4f}")

model.fit(X,y)
train_score = model.score(X,y)
print(f"Training Accuracy: {train_score:.4f}") 
print(f"Validation Accuracy: {np.mean(scores):.4f}") 