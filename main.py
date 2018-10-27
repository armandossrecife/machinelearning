import sys
from datetime import datetime as DT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans

def to_time(str, mask):
    return DT.strptime(str, mask)

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
    """ Cleans and add columns from columns already in the data"""
    # Convert string to datetime and get the 
    # difference in days from beginning to end of the campaign
    df = csvdata
    df['running_days'] = (
              csvdata['deadline'].apply(to_time, args=('%Y-%m-%d',)) 
            - csvdata['launched'].apply(to_time, args=('%Y-%m-%d %H:%M:%S',))
        ).apply(lambda x: x.days)
  
    df, state_dict = to_numeric(df, 'state')
    df, cat_dict = to_numeric(df, 'category')
    df, main_cat_dict = to_numeric(df, 'main_category')
    df, country_dict = to_numeric(df, 'country')

    return df

def kmeans(tdata):
    array_data = tdata[['category','main_category','state','backers','country','usd_pledged_real','usd_goal_real','running_days']].values
    km = KMeans(n_clusters=10, random_state=0)
    km.fit(array_data)

    dc = {}

    for i in km.labels_:
        try:
            dc[i] += 1
        except KeyError:
            dc[i] = 0
            dc[i] += 1

    print(dc)

if __name__ == '__main__':
    # The second arg is the file to get the data from!
    try:
        if len(sys.argv) == 2:    
            data = pd.read_csv(sys.argv[1])
            tdata = transform_data(data)
            kmeans(tdata)

    except Exception as e:
        print(e)