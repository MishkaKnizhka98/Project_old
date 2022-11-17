import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_data():
    #dataset = input("Type the dataset file: ")
    #importing the dataset
    data = pd.read_csv("student-mat.csv")

    #Editing the raw dataset to get x_train and y_train
    #school_name = input("Choose the school (GP or MS): ")

    data = data.loc[data["school"] == "GP"]
    data = data[["sex", "age", "famsize", "Pstatus", "Mjob", "Fjob", "higher", "activities", "G3"]]
    print(data.head())
    #Turning categorical features into numbers

    non_num = data.select_dtypes(include = "object")
    for column in non_num.columns:
        if len(non_num[column].unique()) == 2:
            non_num = non_num.drop([column], axis = 1)
        else:
            non_num[column] = non_num[column].apply(lambda x: column[0].lower() + "_"+x)
            dummies = pd.get_dummies(non_num[column])
            data = pd.concat([data, dummies], axis = 1)
            data = data.drop([column], axis = 1)

    print(data.head(10))

    x_train = data.drop(["G3"], axis = 1)
    x_train = x_train.to_numpy()
    y_train = data["G3"].to_numpy()
    return x_train, y_train








