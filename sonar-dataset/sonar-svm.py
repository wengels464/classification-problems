# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 07:08:17 2022

@author: William Engels
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


input_data = pd.read_csv('sonar.csv')

y = input_data['targets'].copy()
y = LabelEncoder().fit_transform(y)

X_df = input_data.copy()
X_df.drop(['targets'], inplace=True, axis=1)

# Split the train and test data

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=1, stratify=y)

# Scale the features

sc = StandardScaler()
sc.fit(X_df,y)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Define the classifier
# Hyperparameters tuned manually, will learn to automate later...

classifier = SVC(kernel='poly', degree = 3, coef0 = 1.5, random_state = 1)

# Train the model

classifier.fit(X_train_std, y_train)

# Evaluate

y_pred = classifier.predict(X_test_std)

score = accuracy_score(y_test, y_pred)