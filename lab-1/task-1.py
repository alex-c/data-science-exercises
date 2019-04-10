# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 08:54:27 2019

@author: Alex
"""

import pandas as pd
from lxml import objectify

# Get columns from XML

path = '../datasets/iris/data.all.xml'
parsed = objectify.parse(open(path))
root = parsed.getroot()
header = root.column_header

columns = []

for elt in header.getchildren():
    columns.append({'name': elt.get('name'), 'type': elt.get('type')})

names = [c['name'] for c in columns]

# Get data tables

table = pd.read_csv('../datasets/iris/data.all', sep=' ', header=None, names=names)
table_setosa      = table[table['Class'].str.contains('setosa')]
table_versicolora = table[table['Class'].str.contains('versicolor')]
table_virginica   = table[table['Class'].str.contains('virginica')]

# Statistics

stats = []

for name in names:
    if (name == 'Class'):
        continue
    column = table[name]
    stats.append({'min': column.min(), 'max': column.max(), 'mean': column.mean()})

statsdf = pd.DataFrame(stats, index=names[:-1])
print(statsdf)

# Scatter plots

table.plot.scatter(x=names[1], y=names[0], color='DarkBlue');
plot_sepal = table_setosa.plot.scatter(x=names[1], y=names[0], label='Iris setosa', color='Green');
table_versicolora.plot.scatter(ax = plot_sepal, x=names[1], y=names[0], label='Iris versicolora', color='Blue');
table_virginica.plot.scatter(ax = plot_sepal, x=names[1], y=names[0], label='Iris virginica', color='Red');

table.plot.scatter(x=names[3], y=names[2], color='DarkBlue');
plot_petal = table_setosa.plot.scatter(x=names[3], y=names[2], label='Iris setosa', color='Green');
table_versicolora.plot.scatter(ax = plot_petal, x=names[3], y=names[2], label='Iris versicolora', color='Blue');
table_virginica.plot.scatter(ax = plot_petal, x=names[3], y=names[2], label='Iris virginica', color='Red');

# Scatter matrix

color_map = {"Iris-setosa": "Green", 
               "Iris-versicolor": "Blue", 
               "Iris-virginica": "Red"}
              
colors = table["Class"].map(lambda x: color_map.get(x))

pd.plotting.scatter_matrix(table, color=colors, figsize=(9, 9));

# Box plot

table.plot.box()
