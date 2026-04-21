import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\magsi\\Desktop\\AI-ML\\Scripts\\day13-capstone\\data1.csv")

X = df[['level_diff','age_diff','freq_match','win_rate_diff']]
y = df['compatible']

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

feature_names = poly.get_feature_names_out(X.columns)
X_poly_df = pd.DataFrame(X_poly, columns=feature_names)
print(X_poly_df.head())

X_train, X_test, y_train, y_test = train_test_split(X_poly_df, y, test_size = 0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

importances = model.feature_importances_
feature_names = X_poly_df.columns
feat_importances = pd.Series(importances, index=feature_names) 
feat_importances.nlargest(5).plot(kind='barh') 
plt.title("Top 5 Most Important Features") 
plt.show()

