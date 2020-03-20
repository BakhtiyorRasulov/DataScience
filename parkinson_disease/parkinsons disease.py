#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#initialize dataframe
df=pd.read_csv("parkinsons.data")
features = df.loc[:, df.columns!='status'].values[:, 1:]
labels = df.loc[:, 'status'].values
#print(labels[labels==1].shape[0], labels[labels==0].shape[0])

#use MinMaxScaler
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(features)
y = labels

# split model to train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# train model
model=XGBClassifier()
model.fit(x_train, y_train)

# predict
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)
