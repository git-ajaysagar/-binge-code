#Clustering iris dataset using K-means clustering

# -*- coding: utf-8 -*-
'''By Ajay'''

#Importing required libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#Loading datatset 
df=pd.read_csv('F:\\datasets\\Iris.csv')

#Seeing top 10 rows of the data
print(df.head(10))

#Checking the size of the data set
print(df.size)

#Dropping duplicate values if there are any
df.drop_duplicates(inplace=True)

#Checking the size of the dataset again after removing duplicates
print(df.size)

#Checking for any null value in the dataset
print(df.isnull().sum())       #There aren't any null values in this dataset, so we don't have to replace or remove it

#Taking parameters to form clusters
x=df[['SepalWidthCm','PetalWidthCm']]

#Target variable to form clusters
y=df['Species']

#Converting categorical values ('Species') in to numeical values
le=LabelEncoder()
y=le.fit_transform(df['Species'])
print(y)

#Adjusting the plot size
plt.figure(figsize=(10,8))

#Making a subplot
plt.subplot(211)

#Plotting unclustered data points 
plt.scatter(df['SepalWidthCm'], df['PetalWidthCm'],c=y, cmap='gist_rainbow')

#Initiating KMeans clustering algorithm
km = KMeans(n_clusters = 3, n_jobs = 4, random_state=25)

#Fitting parameters using KMeans algorithm
km.fit(x)

#Finding and printing centroids
centers = km.cluster_centers_
print(centers)

#Finding new clustered labels
nl=km.labels_

#Plotting clustered dataset
plt.subplot(212)
plt.scatter(df['SepalWidthCm'],df['PetalWidthCm'],c=nl,cmap='gist_rainbow')
