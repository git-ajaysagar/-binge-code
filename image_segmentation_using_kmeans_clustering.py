#Image segmentation using K-means clustering

# -*- coding: utf-8 -*-
'''By Ajay'''

#Importing required libraries
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Reading image
im=cv.imread('path to image file')

#Converting image from BGR to RGB color space
im=cv.cvtColor(im,cv.COLOR_BGR2RGB)

#Resizing image to make clustering faster
new_size=(800,600)
im2=cv.resize(im,new_size)

#Adjusting the plot size
plt.figure(figsize=(10,8))
 
#Plotting original image
plt.subplot(211)
plt.imshow(im2)

#Feature engineering of image's pixel array for segmentaion
im3=im2/255.0 
print(im3.shape)
im3=im3.reshape(-1,3)
print(im3.shape)

#Initiating a KMeans object with 3 color clusters
kmn=KMeans(n_clusters=3)

#Fitting image pixels using above KMeans object
kmn.fit(im3)

#Finding centroids of clusters
cen=kmn.cluster_centers_

#Grouping pixel data points around centroids to form segments
new_lab=cen[kmn.labels_]

#Reshaping clustered data points to form segmented image
new_lab=new_lab.reshape(im2.shape)

#Plotting segmented image
plt.subplot(212)
plt.imshow(new_lab)

