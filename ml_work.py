#!/usr/bin/env python3
#  coding: utf-8 -*-
"""
Created on Sun Dec 8 14:08:26 2024

@author: abdallah

-*- This is the Main Work for the Exam -*-

!!! before diving into the code we need to inderstand our dataset

!!! as the documentation says in Kaggle this a dataset aims to analyse data related to cardiac features of patients
it provides various information about patients so below is the features of the dataset
    
age: The age of the patient.
sex: Gender of the patient (0: female, 1: male).
cp: Type of chest pain. (in the dataset there are number between 0-3)
trestbps: Resting blood pressure.
chol: Serum cholesterol.
fbs: Fasting blood sugar > 120 mg/dl.
restecg: Resting electrocardiographic results.
thalach: Maximum heart rate achieved.
exang: Exercise induced angina.
oldpeak: ST depression induced by exercise relative to rest
slope: the slope of the peak exercise ST segment
ca:number of major vessels (0-4) colored by flourosopy
hal: 0 = normal; 1 = fixed defect; 2 = reversable defect

and finally there is ony one class label which is target(1 has a cardiac issue 0 not) 
"""

# importing the necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix





df=pd.read_csv('heart.csv')

# check for Nan or Null values in the dataframe
print(df.isnull().values.any())
print(df.isna().values.any())
#  ----------> return False no null||nan values in the dataframe



Y=df["target"]
X=df.drop('target',axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("this is the stat about Logistice regrression")

####### This is the Logistic Regression Model ################

from sklearn.linear_model import LogisticRegression


logistic_regression=LogisticRegression(random_state=42)
logistic_regression.fit(X_train,Y_train)

y_pred_lr = logistic_regression.predict(X_test)


cmlr = confusion_matrix(Y_test, y_pred_lr)

reportlr = classification_report(Y_test, y_pred_lr, output_dict=True)

print(reportlr['accuracy'])
print(reportlr['1']['recall'])
print(reportlr['1']['precision'])
print(reportlr['1']['f1-score'])



print('####### Naive Bayes Model ###############')
from sklearn.naive_bayes import GaussianNB

naive_bayes = GaussianNB()
naive_bayes.fit(X_train, Y_train)
y_pred_naive_bayes = naive_bayes.predict(X_test)

cmnb = confusion_matrix(Y_test, y_pred_naive_bayes)
reportnb = classification_report(Y_test, y_pred_naive_bayes, output_dict=True)

print(reportnb['accuracy'])
print(reportnb['1']['recall'])
print(reportnb['1']['precision'])
print(reportnb['1']['f1-score'])

print('#########################################')

print('####### Support Vector Machine Model ####')
from sklearn.svm import SVC

support_vector_machine = SVC(probability=True, random_state=42)
support_vector_machine.fit(X_train, Y_train)
y_pred_svm = support_vector_machine.predict(X_test)

cmsvm = confusion_matrix(Y_test, y_pred_svm)
reportsvm = classification_report(Y_test, y_pred_svm, output_dict=True)

print(reportsvm['accuracy'])
print(reportsvm['1']['recall'])
print(reportsvm['1']['precision'])
print(reportsvm['1']['f1-score'])

print('#########################################')

print('####### Decision Tree Model #############')
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, Y_train)
y_pred_dt = decision_tree.predict(X_test)

cmdt = confusion_matrix(Y_test, y_pred_dt)
reportdt = classification_report(Y_test, y_pred_dt, output_dict=True)

print(reportdt['accuracy'])
print(reportdt['1']['recall'])
print(reportdt['1']['precision'])
print(reportdt['1']['f1-score'])

print('#########################################')

print('####### Random Forest Model #############')
from sklearn.ensemble import RandomForestClassifier

random_forest_classifier = RandomForestClassifier(random_state=42)
random_forest_classifier.fit(X_train, Y_train)
y_pred_rf = random_forest_classifier.predict(X_test)

cmrf = confusion_matrix(Y_test, y_pred_rf)
reportrf = classification_report(Y_test, y_pred_rf, output_dict=True)

print(reportrf['accuracy'])
print(reportrf['1']['recall'])
print(reportrf['1']['precision'])
print(reportrf['1']['f1-score'])

print('#########################################')



# Plotting model accuracies
model_names = ['Logistic Regression', 'Naive Bayes', 'SVM', 'Decision Tree', 'Random Forest']
accuracies = [
    reportlr['accuracy'],
    reportnb['accuracy'],
    reportsvm['accuracy'],
    reportdt['accuracy'],
    reportrf['accuracy']
]

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color=['blue', 'green', 'orange', 'red', 'purple'])
plt.title('Model Accuracies')
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_accuracies.png')
plt.show()











""" This is some useful refernce that i've used to work this exam ####
https://stackoverflow.com/questions/29530232/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe
https://www.geeksforgeeks.org/pandas-create-test-and-train-samples-from-dataframe/
https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
"""