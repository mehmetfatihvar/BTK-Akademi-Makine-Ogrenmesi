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
#data frame dilimleme(slice)
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

#Numpy dizi(array) dönüşümü
X=x.values
Y=y.values
#Linear regression
#doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)



#polynomal regression
#doğrusal olmayan (nonlinear model) oluşturma
#2.dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
#4.dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg3=PolynomialFeatures(degree=4)
x_poly3=poly_reg3.fit_transform(X)
lin_reg3=LinearRegression()
lin_reg3.fit(x_poly3,y)


plt.scatter(x.values,y.values,color='red')
plt.plot(x,lin_reg.predict(x.values),color='blue')
plt.show()

#Görselleştirme
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()

plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)),color='blue')
plt.show()

#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6]]))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli=sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli=sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg=SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')

plt.show()
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6]]))

from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor()
r_dt.fit(X,Y)
Z=X+0.5
K=X-0.4

plt.scatter(X,Y,color='red')
plt.plot(X,r_dt.predict(X),color='blue')

plt.plot(x,r_dt.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')
print(r_dt.predict([[11]]))
print(r_dt.predict([[6]]))



 