# -*- coding: utf-8 -*-
"""genre_classifyspotify.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xeM2QBXCBS-7y9rUl5Uj9VSMXC8D0OJK
"""

from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

df_2000 = pd.read_csv("Spotify-2000.csv") # change the address accordingly
df_top10s = pd.read_csv(
    "stop10s.csv", engine='pyarrow') #change the address accordingly


len(df_2000["Top Genre"].unique())

df_2000["Top Genre"].value_counts()

df_2000.drop(columns = ['Index', 'Title', 'Artist', 'Year'], inplace = True)

attributes = df_2000.columns[1:]
for attribute in attributes:
    temp = df_2000[attribute]
    for instance in range(len(temp)):
        if(type(temp[instance]) == str):
            df_2000[attribute][instance] = float(temp[instance].replace(',',''))
# check data types using df.dtype

df = df_2000

# first extracting the genre columns
# getting rid of white spaces and turning it all into lower cases
genre = (df["Top Genre"].str.strip()).str.lower()

# function to split the genre column
def genre_splitter(genre):
    result = genre.copy()
    result = result.str.split(" ")
    for i in range(len(result)):
        if (len(result[i]) > 1):
            result[i] = [result[i][1]]
    return result.str.join('')

genre_m1 = genre.copy()
while(max((genre_m1.str.split(" ")).str.len()) > 1):
    genre_m1 = genre_splitter(genre_m1)

len(genre_m1.unique())

genre_m1.value_counts()

unique = genre_m1.unique()
to_remove = []

# genres that have a single instance only will be placed within the to_remove array
for genre in unique:
    if genre_m1.value_counts()[genre] < 10: # 10 was arbitrarily chosen
        to_remove += [genre]
len(to_remove)

df['Top Genre'] = genre_m1
df

df.set_index(["Top Genre"],drop = False, inplace = True)
for name in to_remove:
    type(name)
    df.drop(index = str(name), inplace = True)

df["Top Genre"].value_counts()

"""***Model Creation***"""

train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)
# training set
X_train = train_set.values[:,1:]
y_train = train_set.values[:,0]

# test set
X_test = test_set.values[:,1:]
y_test = test_set.values[:,0]

from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler().fit(X_train)

# Standard Scaler
X_train_ST = standard_scaler.transform(X_train)
X_test_ST = standard_scaler.transform(X_test)

# obtaining all unique classes
unique = np.unique(y_train)

from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
# 1 hot encoding
y_test_1hot = label_binarize(y_test, classes = unique)
y_train_1hot = label_binarize(y_train, classes = unique)

# labelling
y_test_label = LabelEncoder()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier


from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score



# training the models
model_method1 = LogisticRegression(multi_class = 'ovr').fit(X_train_ST, y_train)

# getting predictions
predictions_method1 = model_method1.predict(X_test_ST)

model = RandomForestClassifier(random_state = 42, min_samples_split = 5)
model.fit(X_train, y_train)
pred = model.predict(X_test_ST)

# from sklearn.metrics import confusion_matrix
# print(f1_score(y_test, predictions_method1, labels = unique, average = 'micro' ))


def predict_genre(features_1):
    pred_genre = model.predict(features_1)
    return pred_genre[0]

