# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:31:04 2019

@author: Alex
"""

from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Fetch data
mnist = fetch_mldata('MNIST original')

# Create pandas object that will be used for stats and vizualization
labels = pd.Series(mnist.target).astype('int').astype('category')
df = pd.DataFrame(mnist.data)
num_images = df.shape[1]
df.columns = ['pixel_'+str(x) for x in range(num_images)]

# Display some stats
values = pd.Series(df.values.ravel())
print(" min: {}, \n max: {}, \n mean: {}, \n median: {}, \n most common value: {}".format(values.min(), values.max(), values.mean(), values.median(), values.value_counts().idxmax()))

# Vizualization of 9 random digits
images_to_plot = 9
random_indices = random.sample(range(70000), images_to_plot)
sample_images = df.loc[random_indices, :]
sample_labels = labels.loc[random_indices]
plt.clf()
plt.style.use('seaborn-muted')
fig, axes = plt.subplots(3,3, figsize=(5,5), sharex=True, sharey=True, subplot_kw=dict(adjustable='box-forced', aspect='equal'))
for i in range(images_to_plot):
    subplot_row = i//3 
    subplot_col = i%3  
    ax = axes[subplot_row, subplot_col]
    plottable_image = np.reshape(sample_images.iloc[i,:].values, (28,28))
    ax.imshow(plottable_image, cmap='gray_r')
    ax.set_title('Digit Label: {}'.format(sample_labels.iloc[i]))
    ax.set_xbound([0,28])
plt.tight_layout()
plt.show()

# Preprocess data
data = StandardScaler().fit_transform(mnist.data)

# Split data into training, test and validation sets
train, test, train_lbl, test_lbl = train_test_split(data, mnist.target, test_size=2/7.0, random_state=0)
test, validate, test_lbl, validate_lbl = train_test_split(test, test_lbl, test_size=3.5/7.0, random_state=0)

# Train classifier
classifier = tree.DecisionTreeClassifier()
classifier.fit(train, train_lbl)

# Cross validation scoring (3-fold is default)
scores = cross_val_score(classifier, validate, validate_lbl)
print(scores)
    