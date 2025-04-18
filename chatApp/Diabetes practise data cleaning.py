import pandas as pd 

df = pd.read_csv("/Users/abhinavpatel/Downloads/The Ultimate Cars Dataset 2024.csv", encoding='ISO-8859-1')
print(df.info())
print(df.isnull().sum())
print(df.head())
