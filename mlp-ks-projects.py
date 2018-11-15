### Implementação do Modelo MLP para os dados do Kickstarter
import pandas as pd
from datetime import datetime as dt
import manipuladados
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def to_time(str, mask):
    return dt.strptime(str, mask)

df =  manipuladados.carrega_dados('ks-projects-201801.csv')

# Convert string to datetime and get the difference in days from beginning to end of the campaign
df['running_days'] = (
    df['deadline'].apply(to_time, args=('%Y-%m-%d',))
    - df['launched'].apply(to_time, args=('%Y-%m-%d %H:%M:%S',))
).apply(lambda x: x.days)

df, cat_dict = manipuladados.to_numeric(df, 'category')
df, main_cat_dict = manipuladados.to_numeric(df, 'main_category')
df, state_dict = manipuladados.to_numeric(df, 'state')
df, country_dict = manipuladados.to_numeric(df, 'country')

# Removing unused features
df.drop('ID', axis=1, inplace=True)
df.drop('name', axis=1, inplace=True)
df.drop('deadline', axis=1, inplace=True)
df.drop('launched', axis=1, inplace=True)
df.drop('pledged', axis=1, inplace=True)
df.drop('usd_pledged', axis=1, inplace=True)
df.drop('goal', axis=1, inplace=True)
df.drop('currency', axis=1, inplace=True)

y_column = 'state'
columns_to_normalize = ['running_days', 'backers', 'usd_pledged_real', 'usd_goal_real']
cols_labels = df.columns.values
sidx = np.argsort(cols_labels)
index_columns = sidx[np.searchsorted(cols_labels,columns_to_normalize,sorter=sidx)]
y_index_columns = sidx[np.searchsorted(cols_labels,y_column,sorter=sidx)]

values = df.values
values_shape = values.shape

for j in index_columns:
    max_value = np.max(values[:,j])
    min_value = np.min(values[:,j])
    mean = np.sum(values[:,j]) / values_shape[0]

    i = 0
    while i < values_shape[0]:
        values[i, j] = (values[i, j] - mean) / (max_value - min_value)
        i = i + 1

y = values[:,y_index_columns]
X = np.concatenate((values[:, 0:y_index_columns], values[:,y_index_columns+1:values_shape[1]]), axis=1)


test_x = X[0:379,:]
test_y = y[0:379]

x_train, x_test, y_train, y_test = train_test_split(test_x, test_y, test_size= 0.3, random_state=27)

clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000, alpha=0.001, solver='sgd', verbose=10, random_state=21,tol=0.000000001)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

c = 0
i = 0
while i < y_test.shape[0]:
    if y_test[i] == y_pred[i]:
        c = c + 1
    i = i + 1
print(y_test.shape[0])
print(c)
