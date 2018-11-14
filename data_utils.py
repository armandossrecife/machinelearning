import sys
import pandas as pd
from datetime import datetime as dt
import numpy as np

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

def normalize_data(dataframe, columns_to_normalize):
    cols_labels = dataframe.columns.values
    sidx = np.argsort(cols_labels)
    index_columns = sidx[np.searchsorted(cols_labels,columns_to_normalize,sorter=sidx)]

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

    return cols_labels, values

if __name__ == '__main__':
    # The second arg is the file to get the data from!
    try:
        if len(sys.argv) == 2:    
            data = pd.read_csv(sys.argv[1])
            tdata = transform_data(data)
            labels, n_matrix_data = normalize_data(tdata, ['running_days', 'backers', 'usd_pledged_real', 'usd_goal_real'])
            print(n_matrix_data)
            df = pd.DataFrame(n_matrix_data)
            df.to_csv("normalized_data.csv")

    except Exception as e:
        print(e)