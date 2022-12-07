import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import time

pd.set_option("display.max_columns", None)
df = pd.read_csv("covid_data.csv")

#df.dropna() #no elements missing



#print(df.columns)
#print(df.dtypes)

#Magnus
# df = pd.read_csv("covid_df.csv", low_memory = False)
print(df.shape)
df = df.copy()
df = df.drop(columns= ["MEDICAL_UNIT"])

df.insert(loc=len(df.columns),column='DEATH',value=0)
df["DEATH"] = df["DEATH"].where(df["DATE_DIED"] != "9999-99-99", 1)
df["DEATH"] = df["DEATH"].where(df["DATE_DIED"] == "9999-99-99", 2)
df = df.drop(columns=["DATE_DIED"])

ignore_columns = ["AGE", "CLASIFFICATION_FINAL", "ICU", "INTUBED"]
for feature in df.columns:
    if not(feature in ignore_columns):
        df.drop(df.loc[(df[feature] == 97) | (df[feature] == 99)].index, inplace=True)
    elif feature == "CLASIFFICATION_FINAL":
        df.drop(df.loc[(df[feature] > 3)].index, inplace=True)

for feature in df.columns:
    if feature != "AGE":
        df[feature] = df[feature] == 2
        df[feature] = df[feature].astype(int)

for i in range(1, 9):
    df.insert(loc=len(df.columns), column=f"AGE_GROUP_{i}",value=0)
    df[f"AGE_GROUP_{i}"] = df["AGE"].apply(lambda x: 1 if (x>=(i-1)*15 and x<i*15) else 0)
    print((i-1)*15, i*15)

df = df.drop(columns=["AGE"])
print(df)
