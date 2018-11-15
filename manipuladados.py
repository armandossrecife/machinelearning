import pandas as pd
import numpy as np

################# Carrega dados de um dataframe #################
def carrega_dados(nome_arquivo):
    #Faz a leitura do arquivo e guarda em um Dataframe
    df = pd.read_csv(nome_arquivo, header='infer', sep=',', encoding='utf-8', engine='python')
    return df

################# Dados em Dataframe para X e Y #################
def separa_features_label(df, features, label):
    X_df = df[features] #Apenas as features que podem ser categorizadas em 0 ou 1
    Y_df = df[label]
    return X_df, Y_df

#Gera as colunas dummies das variaveis categoricas de X e Y e retorna Xdummies_df e Ydummies_df
def gera_colunas_dummies(X_df, Y_df):
    Xdummies_df = pd.get_dummies(X_df).astype(int)
    Ydummies_df = pd.get_dummies(Y_df).astype(int)
    return Xdummies_df, Ydummies_df

################## Etapa de transformacao das features e label em Arrays ##################
def transforma_dataframes_dummies_em_arrays(Xdummies_df, Ydummies_df):
    #Transforma Dataframe em Arrays
    X = Xdummies_df.values
    #Y = Ydummies_df.values
    Y  = Ydummies_df['state_successful'] #foi escolhido apenas a caracteristica de sucesso ou não sucesso como label para o resultado da avaliacao do projeto
    return X, Y

if __name__ == '__main__':
    print("Módulo de manipulação de dados do tipo dataframe e transformação em arrays")
