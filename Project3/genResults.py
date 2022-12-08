import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import time

def open_df(filename):
    pd.set_option("display.max_columns", None)
    df = pd.read_csv(filename)
    return df

def add_death(df):
    df = df.drop(columns= ["MEDICAL_UNIT"])

    df.insert(loc=len(df.columns),column='DEATH',value=0)
    df["DEATH"] = df["DEATH"].where(df["DATE_DIED"] != "9999-99-99", 1)
    df["DEATH"] = df["DEATH"].where(df["DATE_DIED"] == "9999-99-99", 2)
    df = df.drop(columns=["DATE_DIED"])
    return df

def filter_df(df, filter):
    for feature in df.columns:
        if not(feature in filter):
            df.drop(df.loc[(df[feature] == 97) | (df[feature] == 99) | (df[feature] == 98)].index, inplace=True)
        elif feature == "CLASIFFICATION_FINAL":
            df.drop(df.loc[(df[feature] > 3)].index, inplace=True)
        elif feature == "PREGNANT":
            df["PREGNANT"].replace(97, 2) #we assume if unspecified --> not pregnant we replace with 2 = not pregnant, in dataset to avoid elminating samples of males as they contain 97-99
            df["PREGNANT"].replace(99, 2)
    
    return df

# print(df.size)
#print(df["PREGNANT"].where(df["PREGNANT"]==1).dropna()) #2754 values which are pregnant
def convert_df_bool(df):
    for feature in df.columns:
        if feature != "AGE":
            df[feature] = (df[feature] == 1) #bool: True if == 2, false if not
            df[feature] = df[feature].astype(int) #convert bool to int: true = 1, false = 0
    return df

def create_age_groups(df, age_groups):
    for i in range(len(age_groups)-1):
        #print(f"age_start {age_groups[i]} agestop {age_groups[i+1]-1}")
        df.insert(loc=len(df.columns), column=f"AGE_GROUP_{i+1}",value=0)
        df[f"AGE_GROUP_{i+1}"] = df["AGE"].apply(lambda x: 1 if (x>age_groups[i] and x<age_groups[i+1]-1) else 0)

    df = df.drop(columns=["AGE"])
    return df

def define_target(df, target_name, target_par):
    df[target_name] = 0
    for par in target_par:
        df[target_name] += df[par]
    
    code = f"df.{target_name} = df.{target_name}.apply(lambda x: 1 if x>0 else 0)"
    exec(code)
    df = df.drop(columns=target_par)
    return df

# df["HIGH_RISK"] = df["DEATH"] + df["INTUBED"] + df["ICU"]
# df.HIGH_RISK = df.HIGH_RISK.apply(lambda x: 1 if x>0 else 0)

# df = df.drop(columns=["DEATH", "INTUBED", "ICU"])

def get_df(filename, filter=["AGE", "CLASIFFICATION_FINAL", "ICU", "INTUBED"],
            age_groups=[0, 18, 30, 40, 50, 65, 75, 85, 121], target_name="HIGH_RISK",
            target_par=["DEATH", "INTUBED", "ICU"]):
    df = open_df(filename)
    df = add_death(df)
    df = filter_df(df, filter)
    df = convert_df_bool(df)
    df = create_age_groups(df, age_groups)
    df = define_target(df, target_name, target_par)
    return df

#print(df.head())

if __name__ == "__main__":
    # ignore_columns = ["AGE", "CLASIFFICATION_FINAL", "ICU", "INTUBED"]
    # age_groups = [0, 18, 30, 40, 50, 65, 75, 85, 121]
    df = get_df("covid_data.csv")
    # run correlation matrix and plot
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    plt.show()