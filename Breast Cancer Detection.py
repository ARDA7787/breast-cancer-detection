# -*- coding: utf-8 -*-
"""
**BREAST CANCER DETECTION**
"""

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the data
from google.colab import files
uploaded = files.upload()
df= pd.read_csv('data.csv')
df.head(7)

#Count the number of rows and columns in the data set
df. shape

#Count the number of empty (NaN, NAN ,na) values in each column
df.isna().sum()

#Drop the column with all missing values
df = df.dropna(axis=1)

#Get the new count of the number of rows and columns
df.shape

#Get a count of the number of Malognant (M) or Benign (B) cell 
df['diagnosis'].value_counts()

#Visualize the count
sns.countplot(df['diagnosis'], label=' count')

#Look at the data types to see which columns need to be encoded
df. dtypes

#Encode the categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)
df.iloc[:,1]

#Create a pair plot
sns.pairplot(df.iloc[:,1:5],hue='diagnosis')

#Print the first 5 rows of the data
df.head(5)

#Get the correlation of the columns
df.iloc[:,1:121].corr()

#Visualize the correlation
plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:12].corr(),annot=True,fmt='.0%')

#Split the data set into independent (X) and dependent (Y) data sets
X = df.iloc[:,2:31].values
Y = df.iloc[:,1].values

#Split the data set into 75% training and 25% testing 
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Scale the data (Feature Scaling)

from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()

X_train = sc.fit_transform(X_train) 
x_test = sc.fit_transform(X_test)

def models(X_train, Y_train):
  #Logistic Regression
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression(random_state=0)
  log.fit(X_train, Y_train)
  #Decision Tree
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
  tree.fit(X_train, Y_train)
  #Random Forest Classifier
  from sklearn. ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy',random_state = 0)
  forest.fit(X_train, Y_train)
  #Print the models acturacy on the training data
  print('[0]Logistic Regression Training Accuracy:', log.score(X_train,Y_train))
  print('[1]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train)) 
  print('[2]Random Forest Classifier Training Accuracy:', forest.score(X_train,Y_train))

  return log, tree, forest

#Getting all of the models
model = models(X_train,Y_train)

#test model accuracy on test data on confusion matrix
from sklearn.metrics import confusion_matrix

for i in range(len(model)):
 cm = confusion_matrix(Y_test, model[0].predict (X_test))

 TP = cm[0][0]
 TN = cm[1][1]
 FN = cm[1][0]
 FP = cm[0][1]

 print(cm)
 print('testing accuracy = ',( TP + TN )/( TP + TN + FN + FP ))
 print()

#Print the prediction of Random FOrest Classifier Model
pred = model[2].predict(X_test)
print(pred)
print()
print(Y_test)

"""

**THANK YOU**
"""
