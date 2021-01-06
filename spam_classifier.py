#Spam classifier
# -*- coding: utf-8 -*-
'''By Ajay'''

#Imporiting required libraries
import string
import pandas as pd 
from nltk.corpus import stopwords
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score

#Reading dataset
mess=pd.read_csv('F:\\datasets\\spam.csv',encoding='latin-1')

#Printing rows and columns of the data
print(mess.shape)

#Removing duplicate data
mess.drop_duplicates(inplace=True)

#Printing number of rows and columns again
print(mess.shape)

#Selecting useful columns only
mess[['tag','message']]=mess[['v1','v2']]
mess=mess[['tag','message']]

#Using label encoder to convert categorical values into numerical values
le=LabelEncoder()
mess['tag']=le.fit_transform(mess['tag'])

#Checking number of spam and non-spam messages
print(mess['tag'].value_counts())

#Checking for Null values in dataset
print(mess.isna().sum())

#Cleaning data (removing punctuation and stop words from texts)
# nltk.download('stopwords')
def cleaning(message):            #Function to clean data
    new_line=[]
    new_words=[]
    for m in message.lower():
        if m not in string.punctuation:
            new_line.append(m)
    new_line=''.join(new_line)
    for n in new_line.split():
        if n not in stopwords.words('english'):
            new_words.append(n)    
    return new_words
  
#Counting the occurence of every unique word in every message
cv=CountVectorizer(analyzer=cleaning)
new_message=cv.fit_transform(mess['message'])

#Splitting the cleaned message into training and testing 
trainx,testx,trainy,testy= train_test_split(new_message,mess['tag'],test_size=0.2,random_state=5)

#Using multinomial Naive Bayes to train model
mnb=MultinomialNB()
mnb.fit(trainx,trainy)

#Making predictions on testing data
pred=mnb.predict(testx)
print(pred)


#Making a confusion matrix for the output of the model
print(confusion_matrix(testy,pred))

#Checking accuracy of the model
print(accuracy_score(testy,pred))


