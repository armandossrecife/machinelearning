import warnings
warnings.filterwarnings("ignore")
import sys
from datetime import datetime as DT
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from random import randint

#Transforma str em data de acordo com mask
def to_time(str, mask):
    return DT.strptime(str, mask)

#Faz um resumo dos dados do dataframe
def view_details(data):
    print("--- Análise dos dados dos projetos Kickstarter de 2018")
    print("Resumo dos dados de entrada")
    print(data.head())
    print(df.shape[0], 'rows and', df.shape[1], 'columns')

df = pd.read_csv('ks-projects-201801.csv', header='infer', sep=',', encoding='utf-8', engine='python')

colunas = ['ID', 'name', 'category', 'main_category', 'currency', 'deadline', 'goal', 'launched', 'pledged', 'state', 'backers', 'country', 'usd_pledged', 'usd_pledged_real', 'usd_goal_real']
colunas_features = ['pledged', 'backers', 'usd_pledged', 'usd_pledged_real', 'usd_goal_real']
colunas_labels = ['category', 'main_category','state']
coluna_state = ['state']

#Gera um dataframe auxiliar para conter apenas as features
d_features = df[colunas_features]
#d_features = df_aux.drop(df_aux.columns[[0, 1, 2, 3, 4, 5, 7,  9, 11]], axis=1)

df['running_days'] = (df['deadline'].apply(to_time, args=('%Y-%m-%d',)) - df['launched'].apply(to_time, args=('%Y-%m-%d %H:%M:%S',))).apply(lambda x: x.days)

#Adiciona a coluna ruuning_days no dataframe d_features
d_features['running_days'] = df['running_days']

#Gera um dataframe auxiliar para conter apenas os labels
d_label = df[coluna_state]
#d_labels = df_aux.drop(df_aux.columns[[0, 1, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]], axis=1)

#3. Dados em Dataframe para X e Y 1000 itens do dataframe
X_df = d_features[:1000]
y_df = d_label[:1000]

#4. Transforma Dataframe em array
X = X_df.values
y = y_df.values

#pega o maior e o menor valor dentro de X
max_X = np.nanmax(X)
min_X = np.nanmin(X)
mean_X = np.nanmean(X)

# aplica um processo de feature scaling
#X = X[i]  / maior valor dos dados das features.
X = X/max_X

#Gera dados randomicos entre 0 e 6 para y (temporário, até categorizar y)
for i in range(0, len(y)):
    y[i] = randint(0, 1)

y = np.asarray(y, dtype=int)

    # Conjunto de dados de treinamento 10000 amostras
X_train = X[:500]
y_train = y[:500]

# Conjunto de dados de testes (378000-360000) amostras
X_test = X[377000:]
y_test = y[377000:]

#5. Cria o modelo Multi-layer Perceptron
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1,solver='sgd', verbose=10, tol=1e-1, random_state=1,learning_rate_init=.1)

mlp.fit(X_train, y_train)
