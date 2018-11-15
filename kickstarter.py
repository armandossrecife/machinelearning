#!/usr/bin/env python
# coding: utf-8

# ## Carregamento e Limpeza dos Dados

# In[ ]:


import pandas as pd
from datetime import datetime as dt


# In[ ]:


df = pd.read_csv('ks-projects-201801.csv')

def to_time(str, mask):
    return dt.strptime(str, mask)

def col_dict(dataframe, col):
    unique_values = dataframe[col].unique()
    dc = {}

    for idx,val in enumerate(unique_values):
        dc[val] = idx

    return dc

def to_numeric(dataframe, col):
    """ Transforms the values of column col to a numeric mapping.
        Returns the transformed dataframe and the dictionary with the mapping."""
    df = dataframe
    dc = col_dict(df, col)
    df[col] = df[col].apply(lambda x: dc[x])

    return df, dc


"""
State feature transformation:

1 - Removing projects with state equals to 'undefined' or 'live'
2 - Change the values to make a binary classification:

successful: 1
failed: 0
canceled: 0
suspended': 0
"""

df.drop(df[(df.state == 'live') | (df.state == 'undefined')].index, inplace=True)
df['state'] = (df['state'] == 'successful').astype(int)


"""Cleans and add columns from columns already in the data"""
# Convert string to datetime and get the 
# difference in days from beginning to end of the campaign

df['running_days'] = (
    df['deadline'].apply(to_time, args=('%Y-%m-%d',)) 
    - df['launched'].apply(to_time, args=('%Y-%m-%d %H:%M:%S',))
).apply(lambda x: x.days)
  
df, cat_dict = to_numeric(df, 'category')
df, main_cat_dict = to_numeric(df, 'main_category')
df, country_dict = to_numeric(df, 'country')

# Removing unused features
df.drop('ID', axis=1, inplace=True)
df.drop('name', axis=1, inplace=True)
df.drop('deadline', axis=1, inplace=True)
df.drop('launched', axis=1, inplace=True)
df.drop('pledged', axis=1, inplace=True)
df.drop('usd_pledged', axis=1, inplace=True)
df.drop('goal', axis=1, inplace=True)
df.drop('currency', axis=1, inplace=True)


# ## Criando conjuntos de treino e teste

# In[ ]:


from sklearn.model_selection import train_test_split
import numpy as np


# In[ ]:


y_column = 'state'
#columns_to_normalize = ['running_days', 'backers', 'usd_pledged_real', 'usd_goal_real']
cols_labels = df.columns.values
sidx = np.argsort(cols_labels)
#index_columns = sidx[np.searchsorted(cols_labels,columns_to_normalize,sorter=sidx)]
y_index_column = sidx[np.searchsorted(cols_labels,y_column,sorter=sidx)]

values = df.values
values_shape = values.shape

# Shuffling lines of the matrix
np.random.shuffle(values)

y = values[:,y_index_column]
X = np.concatenate((values[:, 0:y_index_column], values[:,y_index_column+1:values_shape[1]]), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=27)


# ## Normalização dos dados (média e feature scaling)

# In[ ]:


#import numpy as np
from sklearn.preprocessing import StandardScaler


# In[ ]:


# y_column = 'state'
# columns_to_normalize = ['running_days', 'backers', 'usd_pledged_real', 'usd_goal_real']
# cols_labels = df.columns.values
# sidx = np.argsort(cols_labels)
# index_columns = sidx[np.searchsorted(cols_labels,columns_to_normalize,sorter=sidx)]
# y_index_columns = sidx[np.searchsorted(cols_labels,y_column,sorter=sidx)]

# values = df.values
# values_shape = values.shape
    
# for j in index_columns:
#     max_value = np.max(values[:,j])
#     min_value = np.min(values[:,j])
#     mean = np.sum(values[:,j]) / values_shape[0]

#     i = 0
#     while i < values_shape[0]:
#         values[i, j] = (values[i, j] - mean) / (max_value - min_value)
#         i = i + 1

# y = values[:,y_index_columns]
# X = np.concatenate((values[:, 0:y_index_columns], values[:,y_index_columns+1:values_shape[1]]), axis=1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ## Rede Neural Artificial

# #### Testando convergência do Perceptron para verificar se os dados são linearmente separáveis

# In[ ]:


from sklearn.linear_model import Perceptron


# In[ ]:


perceptron = Perceptron(random_state = 0)
perceptron.fit(X_train, y_train)
predicted = perceptron.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# In[ ]:


cm = confusion_matrix(y_test, predicted)

#plt.clf() 
#plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
# classNames = ['Negative','Positive']
# plt.title('Perceptron Confusion Matrix - Entire Data')
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# tick_marks = np.arange(len(classNames))
# plt.xticks(tick_marks, classNames, rotation=45)
# plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
  
for i in range(2):
    for j in range(2):
        print(str(s[i][j])+" = "+str(cm[i][j]))
        #plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))

# TODO - Apresentar o gráfico da matriz de confusão
#plt.show()


# #### Testando convergência do Multilayer Perceptron

# In[ ]:


from sklearn.neural_network import MLPClassifier


# In[ ]:


# test_X = X[0:500,:]
# test_y = y[0:500]

clf = MLPClassifier(hidden_layer_sizes=(5,5), max_iter=100, alpha=0.01, solver='adam', verbose=True, tol=0.0001)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

c = 0
i = 0
while i < y_test.shape[0]:
    if y_test[i] == y_pred[i]:
        c = c + 1
    i = i + 1
print(y_test.shape[0])
print(c)


# In[ ]:




