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
        df[feature] = (df[feature] == 2) #bool: True if == 2, false if not
        df[feature] = df[feature].astype(int) #convert bool to int: true = 1, false = 0

age_groups = [0, 18, 30, 40, 50, 65, 75, 85, 121]
for i in range(len(age_groups)-1):
    print(f"age_start {age_groups[i]} agestop {age_groups[i+1]-1}")
    df.insert(loc=len(df.columns), column=f"AGE_GROUP_{i+1}",value=0)
    df[f"AGE_GROUP_{i+1}"] = df["AGE"].apply(lambda x: 1 if (x>age_groups[i] and x<age_groups[i+1]-1) else 0)

df = df.drop(columns=["AGE"])

df["HIGH_RISK"] = df["DEATH"] + df["INTUBED"] + df["ICU"]
df.HIGH_RISK = df.HIGH_RISK.apply(lambda x: 1 if x>0 else 0)

df = df.drop(columns=["DEATH", "INTUBED", "ICU"])

print(df.head())
