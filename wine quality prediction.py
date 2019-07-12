# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:28:02 2019

@author: Ajay
"""
#MODEL FOR PREDICTING THE QUALITY OF WINE

#importing required packages and libraries                 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#loading data into pandas dataframe 
data_wine=pd.read_csv("C:\\Users\\Ajay\\Desktop\\winequality.csv")

#dividing data into label and attributes
x=data_wine.drop("quality",axis=1)
y=data_wine.quality

#splitting data into train and test data
np.random.seed(8)
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2)

#checking if data has split succesfully
print(len(test_x),len(test_y))
reg1=LinearRegression()

#training the model 
reg1.fit(train_x,train_y)

#checking slope and intercept
print(reg1.coef_)
print(reg1.intercept_)

#finding coefficients for most optimal factor of our predictive model
c=pd.DataFrame(reg1.coef_,x.columns)
print(c)

#predicting the values for test data
pred1=reg1.predict(test_x)

#comparing predicted and actual values for the quality of wine
pl=pd.DataFrame({"predicted":pred1,"actual":test_y})

#visualising our predicted and actual values to see the difference
pl.plot(kind="bar")
