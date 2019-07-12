# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 01:17:27 2019

@author: Ajay
"""
#PREDICTING PH OF WATER BASED ON THE CARBONATE PRESENT IN IT.
#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#reading data into the program using pandas
water_data=pd.read_excel("C:\\Users\\Ajay\\Desktop\\water.xls") 

#checking our data
print(water_data.head())
print(water_data.describe)

#dividing the data into labels and attributes/features
x=water_carbo=water_data["Y"].values.reshape(-1,1)
print(x.shape)
y=water_ph=water_data["X"].values
print(y.shape)

#plotting our data 
plt.scatter(x,y,color="red")
plt.show()
#splitting the data in 80% training and 20% testing
np.random.seed(7)
trainx,testx,trainy,testy=train_test_split(x,y,test_size=0.2)

#checking train_test dataset
print(trainx)
print("wait")
print(testx)

#training data and plotting the trained data
lr=LinearRegression()
lr.fit(trainx,trainy)
ic=lr.intercept_  #intercept
slp=lr.coef_  #slope
yy=slp*trainx+ic  #fit line
plt.scatter(trainx,trainy,color="orange")             
plt.plot(trainx,yy,color="yellow")
plt.show()

#predicting the data
pred_y=lr.predict(testx)

#comparing and visualizing actual and predicted data
pl=pd.DataFrame({"Actual":testy,"predicted":pred_y})
pl.plot(kind="bar")
