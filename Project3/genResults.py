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
