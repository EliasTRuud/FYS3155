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
df.drop(columns= ["MEDICAL_UNIT"])

df_columns = df.columns
#print(df_columns)
ignore_columns = ["AGE", "CLASIFFICATION_FINAL", "ICU", "INTUBED"]
for feature in df_columns:
    if not(feature in ignore_columns):
        df.drop(df.loc[(df[feature] == 97) | (df[feature] == 99)].index, inplace=True)
    elif feature == "CLASIFFICATION_FINAL":
        df.drop(df.loc[(df[feature] > 3)].index, inplace=True)

df.insert(loc=21,column='DEATH',value=0)
df["DEATH"] = df["DEATH"].where(df["DATE_DIED"] != "9999-99-99", 1)
df["DEATH"] = df["DEATH"].where(df["DATE_DIED"] == "9999-99-99", 2)
df.drop(columns=["DATE_DIED"])

for feature in df_columns:
    if feature != "AGE":
        df[feature] = df[feature] == 2
        df[feature] = df[feature].astype(int)
    elif feature in ignore_columns and feature != "AGE":
        if df[feature] != 2:
            df[feature] = 0
        else:
            df[feature] = 1

print(df)