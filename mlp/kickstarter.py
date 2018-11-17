#!/usr/bin/env python
# coding: utf-8

# ## Carregamento e Limpeza dos Dados

# In[ ]:


import pandas as pd
from datetime import datetime as dt


# In[ ]:

print("Carregando dados...")
df = pd.read_csv('dados/ks-projects-201801.csv')
print("Dados carregado.")

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
print("Ajustando dados...")
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
print("Dados ajustados.")

# ## Criando conjuntos de treino e teste
print("Criando os conjuntos de treino e teste")
# In[ ]:


from sklearn.model_selection import train_test_split
import numpy as np


# In[ ]:

y_column = 'state'
cols_labels = df.columns.values
sidx = np.argsort(cols_labels)
y_index_column = sidx[np.searchsorted(cols_labels,y_column,sorter=sidx)]

values = df.values
values_shape = values.shape

y = values[:,y_index_column]
X = np.concatenate((values[:, 0:y_index_column], values[:,y_index_column+1:values_shape[1]]), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, shuffle=True)

print(X.shape)


# ## Normalização dos dados (média e feature scaling)
print("Normalização dos dados")
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

print("Matriz de Confusão do Perceptron")
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
print("Testando convergência do Multilayer Perceptron")
# In[ ]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import neural_network


# In[ ]:


"""
The number of hidden neurons should be between the size of the input layer and the size of the output layer.
The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
The number of hidden neurons should be less than twice the size of the input layer.
"""

#neural_arch = [(1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13)]
#neural_arch = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)]
#classifiers = []
#reports = []

#for na in neural_arch:
#    classifiers.append(MLPClassifier(hidden_layer_sizes=na, max_iter=500, alpha=0.001, solver='adam', verbose=True, tol=0.000000001))

#for c in classifiers:
#    c.fit(X_train, y_train)
#    y_pred = c.predict(X_test)
#    reports.append(classification_report(y_test, y_pred))

#for r in reports:
#    print(r)

# c = MLPClassifier(hidden_layer_sizes=(5, 5))

c = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=100, alpha=0.01, solver='adam', verbose=True, tol=0.0001)
c.fit(X_train, y_train)
y_pred = c.predict(X_test)


# ### Apresentação de métricas do modelo
print("Apresentação das métricas do modelo MLP")
# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[ ]:


print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[ ]:


### Regressão Logística
print("Regressão Logística")

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics

logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
y_pred = logisticRegr.predict(X_test)

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[ ]:




