# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 08:35:36 2019

@author: Alex
"""

import pandas as pd
from lxml import objectify

# Get columns from XML

path = '../datasets/iris/data_someMissing.all.xml'
parsed = objectify.parse(open(path))
root = parsed.getroot()
header = root.column_header

columns = []

for elt in header.getchildren():
    columns.append({'name': elt.get('name'), 'type': elt.get('type')})

names = [c['name'] for c in columns]

# Get data tables

original = pd.read_csv('../datasets/iris/data.all', sep=' ', header=None, names=names)

table = pd.read_csv('../datasets/iris/data_someMissing.all', sep=' ', header=None, names=names)

table_setosa      = table[table['Class'].str.contains('setosa')]
table_versicolora = table[table['Class'].str.contains('versicolor')]
table_virginica   = table[table['Class'].str.contains('virginica')]

# Count missing values

print((table == '?').sum())

# Deletion

reduced_table = table

for name in names:
    reduced_table = reduced_table[reduced_table[name] != '?']
    if name != "Class":
        reduced_table[name] = reduced_table[name].astype("float64")
reduced_table.reset_index(drop=True, inplace=True)

print("Remaining rows: {}".format(len(reduced_table.index)))

# Deletion results

color_map = {"Iris-setosa": "Green", 
               "Iris-versicolor": "Blue", 
               "Iris-virginica": "Red"}
              
colors = reduced_table["Class"].map(lambda x: color_map.get(x))

pd.plotting.scatter_matrix(reduced_table, color=colors, figsize=(9, 9));

# Stats

stats = []

for name in names:
    if (name == 'Class'):
        continue
    column_original = original[name]
    column_reduced = reduced_table[name]
    stats.append({'mean (reduced)': column_reduced.mean(), 'mean (original)': column_original.mean(), 'difference': column_original.mean()-column_reduced.mean()})

statsdf = pd.DataFrame(stats, index=names[:-1])
print(statsdf)

# Imputation

corrected_table = table.copy()

for name in names:
    if (name == 'Class'):
        continue
    corrected_table[name] = corrected_table[name].replace('?', statsdf['mean (reduced)'][name])
    corrected_table[name] = corrected_table[name].astype("float64")
corrected_table.reset_index(drop=True, inplace=True)

corrected_table_1 = corrected_table[corrected_table['Class'] != '?']

# Stats

stats_corr = []

for name in names:
    if (name == 'Class'):
        continue
    column_original = original[name]
    column_corrected = corrected_table_1[name]
    stats_corr.append({'mean (corrected)': column_corrected.mean(), 'mean (original)': column_original.mean(), 'difference': column_original.mean()-column_corrected.mean()})

stats_corrected = pd.DataFrame(stats_corr, index=names[:-1])
print(stats_corrected)

# Missing class

missing_class = corrected_table[corrected_table['Class'] == '?']
print(missing_class)

means_corrected = []

for name in names:
    if (name == 'Class'):
        continue
    means_corrected.append({'mean (Iris-setosa)': corrected_table_1[corrected_table_1['Class'] == 'Iris-setosa'][name].mean(),
                           'mean (Iris-versicolor)': corrected_table_1[corrected_table_1['Class'] == 'Iris-versicolor'][name].mean(),
                           'mean (Iris-virginica)': corrected_table_1[corrected_table_1['Class'] == 'Iris-virginica'][name].mean()})

means_corrected = pd.DataFrame(means_corrected, index=names[:-1])
print(means_corrected)