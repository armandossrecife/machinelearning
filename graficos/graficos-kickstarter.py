import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Faz a leitura do arquivo e guarda em um Dataframe
def carrega_dados(nome_arquivo):
    dataframe = pd.read_csv(nome_arquivo, header='infer', sep=',', encoding='utf-8', engine='python')
    dataframe.columns = ['ID', 'name', 'category', 'main_category', 'currency', 'deadline', 'goal', 'launched', 'pledged', 'state', 'backers', 'country', 'usd_pledged', 'usd_pledged_real', 'usd_goal_real']
    return dataframe

def limpa_dados(dataframe):
    # get percentage of nulls
    dataframe.isnull().sum()/dataframe.shape[0]
    # Strip white space in column names
    dataframe.columns = [x.strip() for x in dataframe.columns.tolist()]
    dataframe[(dataframe['name'].isnull()) | (dataframe['category'].isnull())]
    # drop all nulls remaining
    dataframe = dataframe.dropna(axis=0, subset=['name', 'category'])
    dataframe.isna().sum()
    # fill null pledged amounts with 0
    dataframe = dataframe.fillna(value=0)
    # Convert string to float
    dataframe.loc[:,'usd_pledged'] = pd.to_numeric(df['usd_pledged'], downcast='float', errors='coerce')
    # convert goal to float
    dataframe['goal'] = pd.to_numeric(dataframe.goal, downcast='float', errors='coerce')
    # Fill null goals with zero
    dataframe = dataframe.fillna(value=0)
    return dataframe

def plota_categorias(titulo, tamanho):
    sns.set_style('darkgrid')
    mains = df.main_category.value_counts().head(tamanho)
    x = mains.values
    y = mains.index
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111)
    ax = sns.barplot(y=y, x=x, orient='h', palette="cool", alpha=0.8)
    plt.title(titulo)
    plt.show()

def plota_subcategorias(titulo, tamanho):
    cats = df.category.value_counts().head(tamanho)
    x = cats.values
    y = cats.index
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111)
    ax = sns.barplot(y=y, x=x, orient='h', palette="winter", alpha=0.8)
    plt.title(titulo)
    plt.show()

def plota_status_projetos(titulo):
    plt.style.use('seaborn-pastel')
    fig, ax = plt.subplots(1, 1, dpi=100)
    explode = [0,0,.1,.2, .4]
    df.state.value_counts().head(5).plot.pie(autopct='%0.2f%%', explode=explode)
    plt.title(titulo)
    plt.ylabel('')
    plt.show()

def plota_histogrma_projeto_success(titulo):
    df_status = df[['status']==1]
    array_dias = df_status.values.reshape(1, len(df_status))
    dias=pd.Series(array_dias[0])
    plt.hist(dias, bins=10)
    plt.ylabel('Objetivo em U$')
    plt.xlabel('Dias')
    plt.title('Campanhas de Sucesso')
    plt.show()

print("Carregando dados do arquivo ks-projects-201801.csv...")
df = carrega_dados('dados/ks-projects-201801.csv')
print("Dados carregados com sucesso. ")

df = limpa_dados(df)
print("Gera gráfico: Categorias das Campanhas do Kickstarter")
plota_categorias('Categorias das Campanhas do Kickstarter', 10)
print("Gera gráfico: Subcategorias ds Campanhas do Kickstarter")
plota_subcategorias('Subcategorias ds Campanhas do Kickstarter', 15)
print("Gera gráfico: Detalhamento dos status dos projetos")
plota_status_projetos('Detalhamento dos status dos projetos')
