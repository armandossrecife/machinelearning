import sys
from datetime import datetime as DT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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

    df = df[['state','backers','usd_pledged_real','usd_goal_real', 'running_days']]

    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)

    return df_normalized, state_dict

# https://stackoverflow.com/questions/19197715/scikit-learn-k-means-elbow-criterion
def elbow(tdata):
    sse = {}

    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tdata)
        data["clusters"] = kmeans.labels_
        sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()

def kmeans(tdata, k):
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(tdata)

    # Nice Pythonic way to get the indices of the points for each corresponding cluster
    return {i: np.where(km.labels_ == i)[0] for i in range(km.n_clusters)}

def analysis(k, df, dc):
    for i in range(0, k):
        cls_df = df.iloc[dc[i]]
        
        meme = cls_df['state'].value_counts()
        print('---- States in cluster {} ----'.format(i))
        print(meme)

def backer_pledge_plot(df, dc):
    markers = {'successful':'+', 'failed':'x'}
  
    for m in markers:
        d = df[df['state'] == dc[m]]
        plt.scatter(d['backers'], d['usd_pledged_real'], s=20, marker=markers[m])
    
    plt.xlabel("Backers")
    plt.ylabel("Dinheiro dado em dólares")
    plt.legend(['sucesso', 'falha'])
    plt.show()

def backer_pledge_plot_k(df, dc, indexes_dict):
    markers = {'successful':'+', 'failed':'x', 'canceled':'x', 'undefined':'x', 'suspended':'x', 'live':'o'}
    colors = ['red','green','blue','orange']

    # para cada cluster
    for c in indexes_dict:
        d = df.iloc[indexes_dict[c]] # os dados desse cluster
        plt.scatter(d['usd_goal_real'], d['usd_pledged_real'], s=2, c=colors[c])
    
    plt.xlabel("Objetivo em dólares")
    plt.ylabel("Dinheiro dado em dólares")
    plt.legend(['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'])
    plt.show()

if __name__ == '__main__':
    # The second arg is the file to get the data from!
    try:
        if len(sys.argv) == 2:
            data = pd.read_csv(sys.argv[1])
            data = data[['state','backers','usd_pledged_real','deadline','launched','usd_goal_real']]
            tdata, state_dict = transform_data(data)
            df_tdata = tdata
            tdata = tdata.values
            # elbow(tdata) # Para ver o gráfico bonito
           
            k = 4
            indexes_dict = kmeans(tdata, k=k)

            print(state_dict)
            
            for i in indexes_dict:
                print(len(indexes_dict[i]))

            analysis(k, data, indexes_dict)
            #backer_pledge_plot(data, state_dict)
            backer_pledge_plot_k(data, state_dict, indexes_dict)


    except Exception as e:
        print(e)