# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 17:47:30 2023

@author: MFV
""" 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
#pd.read_csv("veriler.csv")
veriler=pd.read_csv('maaslar_yeni.csv')
#data frame dilimleme(slice)
x=veriler.iloc[:,2:5]
y=veriler.iloc[:,5:]

print(veriler.drop(['unvan'],axis=1).corr())

#Numpy dizi(array) dönüşümü
X=x.values
Y=y.values
#Linear regression
#doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)


model=sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())

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







#tahminler



print('Poly OLS')
model2=sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli=sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli=sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg=SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)



print('svr ols')
model3=sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())


#decision Tree Regresyonu
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor()
r_dt.fit(X,Y)




print('dt ols')
model4=sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())

#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())


print('rf ols')
model5=sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())

from sklearn.metrics import r2_score
print('Random Forest R2 degeri')
print(r2_score(Y,rf_reg.predict(X)))


#Ozet R2 degerleri
print("---------------------------")
print('Linear R2 degeri')
print(r2_score(Y,lin_reg.predict(X)))

print('Polynomial R2 degeri')
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))

print('SVR R2 degeri')
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

print('Decision Tree R2 degeri')
print(r2_score(Y,r_dt.predict(X)))

print('Random Forest Tree R2 degeri')
print(r2_score(Y,rf_reg.predict(X)))


 