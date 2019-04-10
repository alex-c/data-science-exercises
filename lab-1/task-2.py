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

table = pd.read_csv('../datasets/iris/data_someMissing.all', sep=' ', header=None, names=names)

table_setosa      = table[table['Class'].str.contains('setosa')]
table_versicolora = table[table['Class'].str.contains('versicolor')]
table_virginica   = table[table['Class'].str.contains('virginica')]