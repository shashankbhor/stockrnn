# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 20:46:24 2018

@author: Shashank
"""

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
''' quoting parameter ignores the quotes in the text'''

samples=len(dataset['Review'])

# Cleaning the texts
import re 
import nltk
# Download the stopwords 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,samples):
    # Remove all special character & numbers 
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    # lower the string
    review=review.lower()
    # spliting into words - default splitting with 'space char' 
    review=review.split()
    # removing stopwords 
    review = [word for word in review if not word in set(stopwords.words('english'))] 
    # Removing stem words 
    ps = PorterStemmer()
    review=[ps.stem(word) for word in review]
    review=' '.join(review)
    corpus.append(review)

# Creating Bag of Word model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

## When applying model - follow below steps 
testReview='This is nice place to visit so experience it once'
testReview='This is bad place'
inputForModel=[]  # have to be list
testReview=re.sub('[^a-zA-Z]',' ',testReview)
testReview=testReview.lower().split()
testReview= [word for word in testReview if not word in set(stopwords.words('english'))] 
ps = PorterStemmer()
testReview=[ps.stem(word) for word in testReview]
testReview=' '.join(testReview)
inputForModel.append(testReview)
inputForModel=cv.transform(inputForModel).toarray()
y_pred = classifier.predict(inputForModel)
if y_pred[0]==1:
    print('Positive Review')
else:
    print('Negative Review')
    
