import pandas as pd

df = pd.read_csv('data.csv')
print(df.head())  # Display the first few rows of the DataFrame
print(df.describe())  # Get summary statistics of the DataFrame
print(df['Score'].mean())  # Calculate the mean of the 'score' column