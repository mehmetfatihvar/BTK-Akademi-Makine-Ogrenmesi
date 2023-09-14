# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 17:47:30 2023

@author: MFV
""" 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#pd.read_csv("veriler.csv")
veriler=pd.read_csv('veriler.csv')

x=veriler.drop(['ulke','cinsiyet'],axis=1)
y=veriler['cinsiyet']

#egitim ve test icin bolunmesi 
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logr =LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred=logr.predict(X_test)
print(y_pred)
print(y_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
 