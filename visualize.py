# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 12:02:15 2018

@author: pooja_ranawade
"""

import os
import pandas as pd
import numpy as np

import seaborn.apionly as sns
%matplotlib inline
import matplotlib.pyplot as plt

from collections import Counter
import nltk

data = pd.read_csv('archive.csv',)
# header
header = list(data.columns.values)

# categorywise different dataframe
all_cat = list(data['Category'].unique())

chem_df = data.loc[data['Category'] == 'Chemistry']
eco_df = data.loc[data['Category'] == 'Economics']
lit_df = data.loc[data['Category'] == 'Literature']
med_df = data.loc[data['Category'] == 'Medicine']
peace_df = data.loc[data['Category'] == 'Peace']
phy_df = data.loc[data['Category'] == 'Physics']


# Question 1: Which country has won the most prizes in each category?
max_count = -float('inf')
max_cat = ''
for cat in all_cat:
    current = data.loc[data['Category'] == cat, 'Category'].agg(['count']).iloc[
        0]
    if current > max_count:
        max_cat = cat
        max_count = current
print('Which category has won the most prizes?:', max_cat)


print('Chemistry\n', chem_df['Birth Country'].value_counts())
plt.figure(figsize=(10, 12))
chem_graph = sns.countplot(y='Birth Country', data=chem_df,
                           order=chem_df['Birth Country'].value_counts().index, palette='GnBu_d')
plt.show()

print('Economics\n', eco_df['Birth Country'].value_counts())
plt.figure(figsize=(10, 12))
eco_graph = sns.countplot(y='Birth Country', data=eco_df,
                          order=eco_df['Birth Country'].value_counts().index, palette='GnBu_d')
plt.show()

print('Medicine\n', med_df['Birth Country'].value_counts())
plt.figure(figsize=(10, 12))
med_graph = sns.countplot(y='Birth Country', data=med_df,
                          order=med_df['Birth Country'].value_counts().index, palette='GnBu_d')
plt.show()

print('Physics\n', phy_df['Birth Country'].value_counts())
plt.figure(figsize=(10, 12))
phy_graph = sns.countplot(y='Birth Country', data=phy_df,
                          order=phy_df['Birth Country'].value_counts().index, palette='GnBu_d')
plt.show()

print('Literature\n', lit_df['Birth Country'].value_counts())
plt.figure(figsize=(10, 12))
chem_graph = sns.countplot(y='Birth Country', data=chem_df,
                           order=chem_df['Birth Country'].value_counts().index, palette='GnBu_d')
plt.show()

print('Peace\n', peace_df['Birth Country'].value_counts())
plt.figure(figsize=(10, 12))
peace_graph = sns.countplot(y='Birth Country', data=peace_df,
                            order=peace_df['Birth Country'].value_counts().index, palette='GnBu_d')
plt.show()


# USA dominance
print(data['Birth Country'].value_counts())
plt.figure(figsize=(10, 12))
usa_graph = sns.countplot(y='Birth Country', data=peace_df,
                          order=data['Birth Country'].value_counts().index, palette='GnBu_d')
plt.show()


# What is the gender of a typical Nobel Prize winner?
print(data['Sex'].value_counts())
plt.figure(figsize=(10, 12))
gender_graph = sns.countplot(y='Sex', data=peace_df,
                             order=data['Sex'].value_counts().index, palette='GnBu_d')
plt.show()


male_df = data.loc[data['Sex'] == 'Male']

# The first woman to win the Nobel Prize
female_df = data.loc[data['Sex'] == 'Female']
min_year = female_df.min().iloc[0]
print(female_df.loc[female_df['Year'] == min_year].iloc[0]['Full Name'])

counts = data['Sex'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.show()

import warnings
warnings.filterwarnings('ignore')
data['Birth Year'] = data['Birth Date'].str[
    0:4].replace('nan', 0).apply(pd.to_numeric)
data['Age'] = data['Year'] - data['Birth Year']

bins = [0, 19, 29, 39, 49, 59, 69, 79, 89, 100]
groupNames = ['Teens', '20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s']
data['Age Category'] = pd.cut(data['Age'], bins, labels=groupNames)

chem_df = data.loc[data['Category'] == 'Chemistry']
eco_df = data.loc[data['Category'] == 'Economics']
lit_df = data.loc[data['Category'] == 'Literature']
med_df = data.loc[data['Category'] == 'Medicine']
peace_df = data.loc[data['Category'] == 'Peace']
phy_df = data.loc[data['Category'] == 'Physics']

counts_chem = chem_df['Sex'].value_counts()
pie(counts, labels=counts_chem.index, autopct='%1.1f%%')
show()

chem_age = chem_df['Age Category'].value_counts()
print(chem_age)
plt.pie(chem_age, labels=chem_age.index, autopct='%1.1f%%')
plt.show()


counts_eco = eco_df['Sex'].value_counts()
pie(counts, labels=counts_eco.index, autopct='%1.1f%%')
show()

eco_age = eco_df['Age Category'].value_counts()
print(chem_age)
plt.pie(eco_age, labels=eco_age.index, autopct='%1.1f%%')
plt.show()


counts_lit = lit_df['Sex'].value_counts()
pie(counts, labels=counts_lit.index, autopct='%1.1f%%')
show()

lit_age = lit_df['Age Category'].value_counts()
print(lit_age)
plt.pie(lit_age, labels=lit_age.index, autopct='%1.1f%%')
plt.show()


counts_phy = phy_df['Sex'].value_counts()
pie(counts, labels=counts_phy.index, autopct='%1.1f%%')
show()

phy_age = phy_df['Age Category'].value_counts()
print(phy_age)
plt.pie(phy_age, labels=phy_age.index, autopct='%1.1f%%')
plt.show()


counts_med = med_df['Sex'].value_counts()
pie(counts, labels=counts_med.index, autopct='%1.1f%%')
show()

med_age = med_df['Age Category'].value_counts()
print(med_age)
plt.pie(med_age, labels=med_age.index, autopct='%1.1f%%')
plt.show()


counts_peace = peace_df['Sex'].value_counts()
pie(counts, labels=counts_peace.index, autopct='%1.1f%%')
show()

peace_age = peace_df['Age Category'].value_counts()
print(peace_age)
plt.pie(peace_age, labels=peace_age.index, autopct='%1.1f%%')
plt.show()


sns.jointplot(x="Year",
        y="Age",
        kind='reg',
        data=data)

plt.show()
sns.boxplot(data=data,
         x='Category',
         y='Age')

plt.show()
sns.lmplot('Year','Age',data=data,lowess=True, aspect=2,  line_kws={'color' : 'black'})
plt.show()


# Question 2: What words are most frequently written in the prize motivation?
top_N = 10
stopwords = nltk.corpus.stopwords.words('english')
re_stopwords = r'\b(?:{})\b'.format('|'.join(stopwords))
words = (data['Motivation']
         .str.lower()
         .replace([r'\|', re_stopwords], [' ', ' '], regex=True)
         .str.cat(sep=' ')
         .split()
         )


# generate DF out of Counter
rslt = pd.DataFrame(Counter(words).most_common(top_N),
                    columns=['Word', 'Frequency'])
rslt = rslt[rslt.Word != '"'].reset_index()
del rslt['index']
rslt = rslt.set_index('Word')
print(rslt)
# plot
rslt.plot.bar(rot=0, figsize=(17, 8), width=0.8)


# Repeat laureates
repeat=pd.DataFrame(Counter(data['Full Name']).most_common(),columns=['Full Name','Frequency'])
repeat=repeat.loc[repeat['Frequency']>1].reset_index()
del repeat['index']
repeat=repeat.set_index('Full Name')
print(repeat)

#organization names
c = data['Organization Name'].value_counts()
plt.figure(figsize=(5,12))
UniversitiesGraph = sns.countplot(y="Organization Name", data=data,
              order=c.nlargest(50).index,
              palette='Reds')
plt.show()

