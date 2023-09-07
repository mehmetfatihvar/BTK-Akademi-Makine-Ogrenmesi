# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 17:47:30 2023

@author: MFV
""" 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#pd.read_csv("veriler.csv")
veriler=pd.read_csv('maaslar.csv')

x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]
X=x.values
Y=y.values
#Linear regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(x.values,y.values,color='red')
plt.plot(x,lin_reg.predict(x.values),color='blue')
plt.show()

#polynomal regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(X)
print(x_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()

#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6]]))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
 