import pandas as pd
from collections import Counter

################## Etapa de leitura dos dados do Dataset ##################
#Faz a leitura do arquivo e guarda em um Dataframe
df = pd.read_csv('ks-projects-201801.csv', header='infer', sep=',', encoding='utf-8', engine='python')

#Definicao das colunas de entrada e saida
colunas_features = ['ID', 'name', 'category', 'main_category', 'currency', 'deadline', 'goal', 'launched', 'pledged', 'backers', 'country', 'usd_pledged', 'usd_pledged_real', 'usd_goal_real']
colunas_features_escolhidas = ['category', 'main_category', 'currency', 'country']
coluna_label = ['state']

################## Etapa de separacao das features e label ##################
#Dados em Dataframe para X e Y
X_df = df[colunas_features_escolhidas] #Apenas as features que podem ser categorizadas em 0 ou 1
Y_df = df[coluna_label]

#Imprime os dataframes de X e Y
print(X_df)
print(Y_df)

#Gera as colunas dummies das variaveis categoricas de X e Y
Xdummies_df = pd.get_dummies(X_df).astype(int)
Ydummies_df = pd.get_dummies(Y_df).astype(int)

################## Etapa de transformacao das features e label em Arrays ##################
#Transforma Dataframe em Arrays
X = Xdummies_df.values
#Y = Ydummies_df.values
Y  = Ydummies_df['state_successful'] #foi escolhido apenas a caracteristica de sucesso ou não sucesso como label para o resultado da avaliacao do projeto

#Imprime os arrays de X e Y
print(X)
print(Y)

################## Etapa de definicao de Treino e Teste ##################
#Define percentual de treino
porcentagem_treino = 0.9

#Define os tamanhos de treino e teste
tamanho_de_treino = int(porcentagem_treino * len(Y))
tamanho_de_teste = len(Y) - tamanho_de_treino

#Dados de treino
treino_dados = X[:tamanho_de_treino] #Conjunto de dados das features
treino_marcacoes = Y[:tamanho_de_treino] #Conjunto de dados dos labels

#Dados de teste
teste_dados = X[-tamanho_de_teste:] #Conjunto de dados das features
teste_marcacoes = Y[-tamanho_de_teste:] #Conjunto de dados dos labels

################## Etapa de checagem de taxa de acerto base, com base nos valores da coluna state_successful ##################
# a eficácia do algoritmo que chuta tudo 0 ou 1
acerto_de_um = sum(Y)
acerto_de_zero = len(Y) - acerto_de_um
#acerto_base = max(Counter(Y).values())
taxa_de_acerto_base = 100.0 * max(acerto_de_um, acerto_de_zero) / len(Y)
#taxa_de_acerto_base = 100.0 * acerto_base / len(Y)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

################## Etapa de definicao do Modelo de Machine Learning ##################
#Importa o modelo baseado no Naive Bayes
from sklearn.naive_bayes import MultinomialNB
#Cria o modelo
modelo = MultinomialNB()
#Ajusta o modelo
modelo.fit(treino_dados, treino_marcacoes)

################## Etapa de Predicao do Modelo baseado nos dados de testes ##################
#Faz a previsão baseado nos dados de testes
resultado = modelo.predict(teste_dados)

################## Etapa do calculo da taxa de acerto do Modelo MultinomialNB baseado nos dados de testes ##################
#Gera o vetor de acertos
diferencas = resultado - teste_marcacoes

#Faz o calculo da taxa de acertos
acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print("Taxa de acerto do MultinomialNB: ", taxa_de_acerto)
print("Total elementos do teste: ", total_de_elementos)
