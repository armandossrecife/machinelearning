from sklearn.naive_bayes import MultinomialNB #Importa o modelo baseado no Naive Bayes

# define porcentagem_treino, tamanho_de_treino, tamanho_de_teste
def percentual_treino(percentual, Y):
    #Define percentual de treino
    porcentagem_treino = percentual
    #Define os tamanhos de treino e teste
    tamanho_de_treino = int(porcentagem_treino * len(Y))
    tamanho_de_teste = len(Y) - tamanho_de_treino
    return porcentagem_treino, tamanho_de_treino, tamanho_de_teste

# define dados de treino e teste
def dados_treino_teste(X, Y, tamanho_de_treino, tamanho_de_teste):
    #Dados de treino
    treino_dados = X[:tamanho_de_treino] #Conjunto de dados das features
    treino_marcacoes = Y[:tamanho_de_treino] #Conjunto de dados dos labels

    #Dados de teste
    teste_dados = X[-tamanho_de_teste:] #Conjunto de dados das features
    teste_marcacoes = Y[-tamanho_de_teste:] #Conjunto de dados dos labels
    return treino_dados, treino_marcacoes, teste_dados, teste_marcacoes

################## Etapa de definicao do Modelo de Machine Learning ##################
# Cria modelo MultinomialNB
def cria_modelo_treino_multinomialnb(treino_dados, treino_marcacoes):
    #Cria o modelo
    modelo = MultinomialNB()
    #Ajusta o modelo
    modelo.fit(treino_dados, treino_marcacoes)
    return modelo

################## Etapa de Predicao do Modelo baseado nos dados de testes ##################
# Faz a predição do modelo baseado nos dados de teste
def predicao_modelo_multinomialnb(modelo, teste_dados):
    resultado = modelo.predict(teste_dados)
    return resultado

if __name__ == '__main__':
    print("Módulo de manipulação de modelos de machine learning")
