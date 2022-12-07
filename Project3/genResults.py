import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import time

df = pd.read_csv("covid_data.csv")
df.insert(loc=21,column='DEATH',value=0)

df["DEATH"] = df["DEATH"].where(df["DATE_DIED"] != "9999-99-99", 1)
df["DEATH"] = df["DEATH"].where(df["DATE_DIED"] == "9999-99-99", 2)

df.drop(columns=["DATE_DIED"])
#df.dropna() #no elements missing


df["AT_RISK"] =  0
print(df.head())
#print(df.columns)
#print(df.dtypes)

#Magnus
pd.set_option("display.max_columns", None)
data = pd.read_csv("covid_data.csv", low_memory = False)
print(data.shape)
data1 = data.copy()
data1.drop(columns= ["MEDICAL_UNIT"])

data1_columns = data1.columns
print(data1_columns)
ignore_columns = ["DATE_DIED", "AGE", "CLASIFFICATION_FINAL", "ICU", "INTUBED", "MEDICAL_UNIT"]
for feature in data1_columns:
    if not(feature in ignore_columns):
        data1.drop(data1.loc[(data1[feature] == 97) | (data1[feature] == 99)].index, inplace=True)
    elif feature == "CLASIFFICATION_FINAL":
        data1.drop(data1.loc[(data1[feature] > 3)].index, inplace=True)


print(data1.shape)
