import pandas as pd

df = pd.read_csv(r'C:\Users\magsi\Desktop\AI-ML\Scripts\day4\data.csv')

print(df.head())

print("missing values: \n ", df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].mean())

df['Score'] = df['Score'].fillna(0)

print("cleaned data: \n ", df)