# -*- coding: utf-8 -*-

import sys
import pandas as pd
from datetime import datetime as dt
import numpy as np

# MLP
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


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
    # Isso pode ser feito com o proprio pandas
    # TODO: refatorar
    df = dataframe
    dc = col_dict(df, col)
    df[col] = df[col].apply(lambda x: dc[x])

    return df, dc

def transform_data(csvdata):
    df = csvdata

    """ Cleans and add columns from columns already in the data"""
    # Convert string to datetime and get the 
    # difference in days from beginning to end of the campaign

    df['running_days'] = (
              csvdata['deadline'].apply(to_time, args=('%Y-%m-%d',)) 
            - csvdata['launched'].apply(to_time, args=('%Y-%m-%d %H:%M:%S',))
        ).apply(lambda x: x.days)
  
    df, cat_dict = to_numeric(df, 'category')
    df, main_cat_dict = to_numeric(df, 'main_category')
    df, state_dict = to_numeric(df, 'state')
    df, country_dict = to_numeric(df, 'country')
    #df, country_dict = to_numeric(df, 'currency')

    # Removing unused features
    df = df.drop('ID', 1)
    df = df.drop('name', 1)
    df = df.drop('deadline', 1)
    df = df.drop('launched', 1)
    df = df.drop('pledged', 1)
    df = df.drop('usd_pledged', 1)
    df = df.drop('goal', 1)
    df = df.drop('currency', 1)
    
    return df

def normalize_data(dataframe, columns_to_normalize, y_column):
    cols_labels = dataframe.columns.values
    sidx = np.argsort(cols_labels)
    index_columns = sidx[np.searchsorted(cols_labels,columns_to_normalize,sorter=sidx)]
    y_index_columns = sidx[np.searchsorted(cols_labels,y_column,sorter=sidx)]

    values = dataframe.values
    values_shape = values.shape
    
    for j in index_columns:
        max = np.max(values[:,j])
        min = np.min(values[:,j])
        mean = np.sum(values[:,j]) / values_shape[0]

        i = 0
        while i < values_shape[0]:
            values[i, j] = (values[i, j] - mean) / (max - min)
            i = i + 1

    y = values[:,y_index_columns]
    x = np.concatenate((values[:, 0:y_index_columns], values[:,y_index_columns+1:values_shape[1]]), axis=1)

    return cols_labels, x, y

if __name__ == '__main__':
    # The second arg is the file to get the data from!
    try:
        if len(sys.argv) == 2:    
            data = pd.read_csv(sys.argv[1])
            tdata = transform_data(data)
            labels, x, y = normalize_data(tdata, ['running_days', 'backers', 'usd_pledged_real', 'usd_goal_real'], 'state')

            print("Iniciando treinamento da MLP")

            #['category' 'main_category' 'state' 'backers' 'country' 'usd_pledged_real' 'usd_goal_real' 'running_days']

            # Uma fração da matriz
            test_x = x[0:379,:]
            test_y = y[0:379]

            # Testing the MLP
            x_train, x_test, y_train, y_test = train_test_split(test_x, test_y, test_size= 0.3, random_state=27)
            # clf = MLPClassifier(hidden_layer_sizes=(7, 50, 1), max_iter=1000, alpha=1, solver='sgd', verbose=10, random_state=21,tol=0.000000001)
            clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000, alpha=0.001, solver='sgd', verbose=10, random_state=21,tol=0.000000001)
            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)

            print("x_test: ", x_test)
            print("Predict de x_test", y_pred)

            c = 0
            i = 0
            while i < y_test.shape[0]:
                if y_test[i] == y_pred[i]:
                    c = c + 1
                i = i + 1
            print(y_test.shape[0])
            print(c)

            # accuracy_score(y_test, y_pred, normalize=False)

    except Exception as e:
        print(e)