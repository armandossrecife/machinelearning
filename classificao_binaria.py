import manipuladados
import manipulamodelos
from collections import Counter

#Definicao das colunas de entrada e saida
colunas_features = ['ID', 'name', 'category', 'main_category', 'currency', 'deadline', 'goal', 'launched', 'pledged', 'backers', 'country', 'usd_pledged', 'usd_pledged_real', 'usd_goal_real']
colunas_features_escolhidas = ['category', 'main_category', 'currency', 'country']
coluna_label = ['state']

################# Carrega dados de um dataframe #################
dataframe = manipuladados.carrega_dados('ks-projects-201801.csv')

################# Dados em Dataframe para X_df e Y_df #################
X_df, Y_df = manipuladados.separa_features_label(dataframe, colunas_features_escolhidas, coluna_label)

#Gera as colunas dummies das variaveis categoricas de X_df e Y_df e devolve Xdummies_df, Ydummies_df
Xdummies_df, Ydummies_df = manipuladados.gera_colunas_dummies(X_df, Y_df)

################## Etapa de transformacao das features e label em Arrays X e Y ##################
X, Y = manipuladados.transforma_dataframes_dummies_em_arrays(Xdummies_df, Ydummies_df)

################## Etapa de checagem de taxa de acerto base, com base nos valores da coluna state_successful ##################
# a eficácia do algoritmo que chuta tudo 0 ou 1
acerto_de_um = sum(Y)
acerto_de_zero = len(Y) - acerto_de_um
taxa_de_acerto_base = 100.0 * max(acerto_de_um, acerto_de_zero) / len(Y)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

# define porcentagem_treino, tamanho_de_treino, tamanho_de_teste
porcentagem_treino, tamanho_de_treino, tamanho_de_teste = manipulamodelos.percentual_treino(0.9, Y)

# define dados de treino e teste
treino_dados, treino_marcacoes, teste_dados, teste_marcacoes = manipulamodelos.dados_treino_teste(X, Y, tamanho_de_treino, tamanho_de_teste)

# Cria modelo treinado do tipo MultinomialNB
modelo_treinado_multinomialnb = manipulamodelos.cria_modelo_treino_multinomialnb(treino_dados, treino_marcacoes)

# Faz a predição do modelo baseado nos dados de teste
resultado = manipulamodelos.predicao_modelo_multinomialnb(modelo_treinado_multinomialnb, teste_dados)

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
