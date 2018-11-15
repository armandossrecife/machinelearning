import matplotlib.pyplot as plt
import sklearn
import pandas as pd
#from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

# Load data from https://www.openml.org/d/554
#1. Faz a leitura do arquivo e guarda em um Dataframe
df = pd.read_csv('mnist_784.csv')

#2. Define as colunas
colunas = df.columns
colunas_features = colunas[0:784]
colunas_label = colunas[784]

#3. Dados em Dataframe para X e Y
X_df = df[colunas_features]
y_df = df[colunas_label]

#4. Transforma Datafram em array
X = X_df.values
y = y_df.values

# aplica um processo de feature scaling
X = X / 255.

# Conjunto de dados de treinamento 60000 amostras
X_train = X[:60000]
y_train = y[:60000]

# Conjunto de dados de testes 10000 amostras
X_test = X[60000:]
y_test = y[60000:]

#5. Cria o modelo Multi-layer Perceptron
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,solver='sgd', verbose=10, tol=1e-4, random_state=1,learning_rate_init=.1)

#6. Ajusta o modelo aos dados de treino
mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

y_prediction = mlp.predict(X_test)

print("X testes ", X_test)
print("Y predictions: ", y_prediction)
