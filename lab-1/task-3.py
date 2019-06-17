# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:31:04 2019

@author: Alex
"""

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree

# Fetch data
mnist = fetch_mldata('MNIST original')

# Preprocess data
data = StandardScaler().fit_transform(mnist.data)
# Split data into training, test and validation sets
train, test, train_lbl, test_lbl = train_test_split(data, mnist.target, test_size=2/7.0, random_state=0)
test, validate, test_lbl, validate_lbl = train_test_split(test, test_lbl, test_size=3.5/7.0, random_state=0)
print(train.data.shape)
print(test.data.shape)
print(validate.data.shape)

# Train classifier
classifier = tree.DecisionTreeClassifier()
classifier.fit(train, train_lbl)